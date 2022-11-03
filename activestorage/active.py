import os
import numpy as np

from math import prod
from netCDF4 import Dataset

from zarr.indexing import (
    OrthogonalIndexer,
)
from activestorage.storage import reduce_chunk
from activestorage import netcdf_to_zarr as nz


class Active:
    """ 
    Instantiates an interface to active storage which contains either zarr files
    or HDF5 (NetCDF4) files.

    This is Verson 1 which simply provides support for standard read operations, but done via
    explicit reads within this class rather than within the underlying format libraries.
    
    Version 2 will add methods for actual active storage.

    """
    def __new__(cls, *args, **kwargs):
        """Store reduction methods."""
        instance = super().__new__(cls)
        instance._methods = {
            "min": np.min,
            "max": np.max,
            "sum": np.sum,
            # For the unweighted mean we calulate the sum and divide
            # by the number of non-missing elements
            "mean": np.sum,
        }
        return instance

    def __init__(self, uri, ncvar=None):
        """Instantiate in the same way as normal.

        :Parameters:

            ncvar: `str`, optional
                The netCDF variable name of the data. May also be set
                with the `ncvar` attribute.

        """
        # Assume NetCDF4 for now
        self.uri = uri
        if self.uri is None:
            raise ValueError(f"Must use a valid file for uri. Got {self.uri}")
        if not os.path.isfile(self.uri):
            raise ValueError(f"Must use existing file for uri. {self.uri} not found")
        self.ncvar = ncvar
        self.zds = None

        self._version = 1
        self._components = False
        self._method = None

    def __getitem__(self, index):
        """ 
        Provides support for a standard get item.
        """
        # In version one this is done by explicitly looping over each chunk in the file
        # and returning the requested slice ourselves. In version 2, we can pass this
        # through to the default method.
        ncvar = self.ncvar
        if ncvar is None:
            raise ValueError("Must set a netCDF variable name to slice")

        if self.method is None and self._version == 0:
            # No active operation
            nc = Dataset(self.uri)
            data = nc[ncvar][index]
            nc.close()
            return data
        elif self._version == 1:
            return self._via_kerchunk(index)
        elif self._version  == 2:
            return self._via_kerchunk(index)
        else:
            raise ValueError(f'Version {self._version} not supported')

    @property
    def components(self):
        """Return or set the components flag.

        If True and `method` is not `None` then return the processed
        result in a dictionary that includes a processed value and the
        sample size, from which the final result can be calculated.

        """
        return self._components

    @components.setter
    def components(self, value):
        self._components = bool(value)

    @property
    def method(self):
        """Return or set the active method.

        The active method to apply when retrieving a subspace of the
        data. By default the data is returned unprocessed. Valid
        methods are:

        ==========  ==================================================
        *method*    Description
        ==========  ==================================================
        ``'min'``   The minumum

        ``'max'``   The maximum

        ``'mean'``  The unweighted mean

        ``'sum'``   The sum
        ==========  ==================================================

        """
        return self._methods.get(self._method)

    @method.setter
    def method(self, value):
        if value is not None and value not in self._methods:
            raise ValueError(f"Bad 'method': {value}. Choose from min/max/mean/sum.")

        self._method = value

    @property
    def ncvar(self):
        """Return or set the netCDF variable name."""
        return self._ncvar

    @ncvar.setter
    def ncvar(self, value):
        self._ncvar = value

    def _get_active(self, method, *args):
        """ 
        *args defines a slice of data. This method loops over each of the chunks
        necessary to extract the parts of the slice, and asks the active storage 
        to apply the method to each part. It then applies the method to 
        the partial results and returns a value is if  method had been applied to
        an array returned via getitem.
        """
        raise NotImplementedError

    def _via_kerchunk(self, index):
        """ 
        The objective is to use kerchunk to read the slices ourselves. 
        """
        # FIXME: Order of calls is hardcoded'
        if self.zds is None:
            ds = nz.load_netcdf_zarr_generic(self.uri, self.ncvar)
            # The following is a hangove from exploration
            # and is needed if using the original doing it ourselves
            # self.zds = make_an_array_instance_active(ds)
            self.zds = ds

        return self._get_selection(index)

    def _get_selection(self, *args):
        """ 
        First we need to convert the selection into chunk coordinates,
        steps etc, via the Zarr machinery, then we get everything else we can
        from zarr and friends and use simple dictionaries and tupes, then
        we can go to the storage layer with no zarr.
        """
        compressor = self.zds._compressor
        filters = self.zds._filters

        # FIXME: populate this from metadata, see issue #18
        # interpretation: (_fillvalue, missing, min_valid_value, max_valid_value)
        missing = (None, None, None, None)  # FIXME: Needs implementation 


        indexer = OrthogonalIndexer(*args, self.zds)
        out_shape = indexer.shape
        out_dtype = self.zds._dtype
        stripped_indexer = [(a, b, c) for a,b,c in indexer]
        drop_axes = indexer.drop_axes  # not sure what this does and why, yet.

        # yes this next line is bordering on voodoo ... 
        fsref = self.zds.chunk_store._mutable_mapping.fs.references

        return self._from_storage(stripped_indexer, drop_axes, out_shape,
                                  out_dtype, compressor, filters, missing, fsref)

    def _from_storage(self, stripped_indexer, drop_axes, out_shape, out_dtype,
                      compressor, filters, missing, fsref):
        method = self.method
        if method is not None:
            out = []
        else:
            out = np.empty(out_shape, dtype=out_dtype, order=self.zds._order)

        for chunk_coords, chunk_selection, out_selection in stripped_indexer:
            self._process_chunk(fsref, chunk_coords,chunk_selection,
                                out, out_selection,
                                compressor, filters, missing,
                                drop_axes=drop_axes)

        if method is not None:
            # Apply the method (again) to aggregate the result
            out = method(out)
            
            if self._components:
                # Return a dictionary of components containing the
                # reduced data and the sample size ('n'). (Rationale:
                # cf-python needs the sample size for all reductions;
                # see the 'mtol' parameter of cf.Field.collapse.)
                #
                # Note that in this case the reduced data must always
                # have the same number of dimensions as the original
                # array, i.e. 'drop_axes' is always considered False,
                # regardless of its setting. (Rationale: dask
                # reductions require the per-dask-chunk partial
                # reductions to retain these dimensions so that
                # partial results can be concatenated correctly.)
                n = prod(out_shape)
                shape1 = (1,) * len(out_shape)
                n = np.reshape(n, shape1)
                out = out.reshape(shape1)

                if self._method == "mean":
                    # For the average, the returned component is
                    # "sum", not "mean"
                    out = {"sum": out, "n": n}
                else:
                    out = {self._method: out, "n": n}
            else:
                # Return the reduced data as a numpy array. For most
                # methods the data is already in this form.
                if self._method == "mean":
                    # For the average, it is actually the sum that has
                    # been created, so we need to divide by the sample
                    # size.
                    n = prod(out_shape)
                    out = out / n

        return out

    def _process_chunk(self, fsref, chunk_coords, chunk_selection, out,
                       out_selection, compressor, filters, missing, 
                       drop_axes=None):
        """Obtain part or whole of a chunk.

         This is done by taking binary data from storage and filling
         the output array.

        """
        coord = '.'.join([str(c) for c in chunk_coords])
        key = f"{self.ncvar}/{coord}"
        rfile, offset, size = tuple(fsref[key])
        tmp = reduce_chunk(rfile, offset, size, compressor, filters, missing,
                           self.zds._dtype, self.zds._chunks, self.zds._order,
                           chunk_selection, method=self.method)

        if self.method is not None:
            out.append(tmp)
        else:

            if drop_axes:
                tmp = np.squeeze(tmp, axis=drop_axes)

            # store selected data in output
            out[out_selection] = tmp
