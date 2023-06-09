import os
import numpy as np
import pathlib

import s3fs
import xarray as xr

#FIXME: Consider using h5py throughout, for more generality
from netCDF4 import Dataset
from zarr.indexing import (
    OrthogonalIndexer,
)
from activestorage.config import *
from activestorage.s3 import reduce_chunk as s3_reduce_chunk
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

    def __init__(self, uri, ncvar, storage_type=None, missing_value=None, _FillValue=None, valid_min=None, valid_max=None):
        """
        Instantiate with a NetCDF4 dataset and the variable of interest within that file.
        (We need the variable, because we need variable specific metadata from within that
        file, however, if that information is available at instantiation, it can be provided
        using keywords and avoid a metadata read.)
        """
        # Assume NetCDF4 for now
        self.uri = uri
        if self.uri is None:
            raise ValueError(f"Must use a valid file for uri. Got {self.uri}")
        self.storage_type = storage_type
        if self.storage_type == "s3":
            USE_S3 = True
        if not os.path.isfile(self.uri) and not self.storage_type:
            raise ValueError(f"Must use existing file for uri. {self.uri} not found")
        self.ncvar = ncvar
        if self.ncvar is None:
            raise ValueError("Must set a netCDF variable name to slice")
        self.zds = None

        self._version = 1
        self._components = False
        self._method = None
        self._lock = False
      
        # obtain metadata, using netcdf4_python for now
        # FIXME: There is an outstanding issue with ._FilLValue to be handled.
        # If the user actually wrote the data with no fill value, or the
        # default fill value is in play, then this might go wrong.
        if (missing_value, _FillValue, valid_min, valid_max) == (None, None, None, None):
            if storage_type is None:
                ds = Dataset(uri)
            elif storage_type == "s3":
                # correct settings for Minio; need be imported from config.py
                # calling open returns a File-like object S3FileSystem
                fs = s3fs.S3FileSystem(key="minioadmin",
                                       secret="minioadmin",
                                       client_kwargs={'endpoint_url': "http://localhost:9000"})
                with fs.open(uri, 'rb') as s3file:
                    # this will throw a FileNotFoundError: [Errno 2] No such file or directory: '<File-like object S3FileSystem, pyactivestorage/s3_test_bizarre.nc>'
                    # ds = Dataset(s3file)
                    # try use xarray for now
                    ds = xr.open_dataset(s3file, engine='netcdf4')
            try:
                ds_var = ds[ncvar]
            except IndexError as exc:
                print(f"Dataset {ds} does not contain ncvar {ncvar!r}.")
                raise exc

            self._filters = ds_var.filters()
            self._missing = getattr(ds_var, 'missing_value', None)
            self._fillvalue = getattr(ds_var, '_FillValue', None)
            valid_min = getattr(ds_var, 'valid_min', None)
            valid_max = getattr(ds_var, 'valid_max', None)
            valid_range = getattr(ds_var, 'valid_range', None)
            if valid_max is not None or valid_min is not None:
                if valid_range is not None:
                    raise ValueError(
                        "Invalid combination in the file of valid_min, "
                        "valid_max, valid_range: "
                        f"{valid_min}, {valid_max}, {valid_range}"
                    )                
                valid_range = (valid_min, valid_max)
            else:
                valid_range = (None, None)
            self._valid_min, self._valid_max = valid_range
            
            ds.close()
        else:
            self._missing = missing_value
            self._fillvalue = _FillValue
            self._valid_min = valid_min
            self._valid_max = valid_max

    def __getitem__(self, index):
        """ 
        Provides support for a standard get item.
        """
        # In version one this is done by explicitly looping over each chunk in the file
        # and returning the requested slice ourselves. In version 2, we can pass this
        # through to the default method.
        ncvar = self.ncvar

        if self.method is None and self._version == 0:
            # No active operation
            lock = self.lock
            if lock:
                lock.acquire()
                
            nc = Dataset(self.uri)
            data = nc[ncvar][index]
            nc.close()

            if lock:
                lock.release()

            return data
        elif self._version == 1:
            return self._via_kerchunk(index)
        elif self._version  == 2:
            # No active operation either
            lock = self.lock
            if lock:
                lock.acquire()

            data = self._via_kerchunk(index)

            if lock:
                lock.release()

            return data
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

        ``'sum'``   The unweighted sum
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

    @property
    def lock(self):
        """Return or set a lock that prevents concurrent file reads when accessing the data locally.

        The lock is either a `threading.Lock` instance, an object with
        same API and functionality (such as
        `dask.utils.SerializableLock`), or is `False` if no lock is
        required.

        To be effective, the same lock instance must be used across
        all process threads.

        """
        return self._lock

    @lock.setter
    def lock(self, value):
        if not value:
            value = False
            
        self._lock = value

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
        from zarr and friends and use simple dictionaries and tuples, then
        we can go to the storage layer with no zarr.
        """
        compressor = self.zds._compressor
        filters = self.zds._filters

        missing = self._fillvalue, self._missing, self._valid_min, self._valid_max

        indexer = OrthogonalIndexer(*args, self.zds)
        out_shape = indexer.shape
        out_dtype = self.zds._dtype
        stripped_indexer = [(a, b, c) for a,b,c in indexer]
        drop_axes = indexer.drop_axes  # not sure what this does and why, yet.

        # yes this next line is bordering on voodoo ...
        # this returns a nested dictionary with the full file FS reference
        # ie all the gubbins: chunks, data structure, types, etc
        # if using zarr<=2.13.3 call with _mutable_mapping ie
        # fsref = self.zds.chunk_store._mutable_mapping.fs.references 
        fsref = self.zds.chunk_store.fs.references

        return self._from_storage(stripped_indexer, drop_axes, out_shape,
                                  out_dtype, compressor, filters, missing, fsref)

    def _from_storage(self, stripped_indexer, drop_axes, out_shape, out_dtype,
                      compressor, filters, missing, fsref):
        method = self.method
        if method is not None:
            out = []
            counts = []
        else:
            out = np.empty(out_shape, dtype=out_dtype, order=self.zds._order)
            counts = None  # should never get touched with no method!

        for chunk_coords, chunk_selection, out_selection in stripped_indexer:
            self._process_chunk(fsref, chunk_coords,chunk_selection,
                                out, counts, out_selection,
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
                n = np.prod(out_shape)
                shape1 = (1,) * len(out_shape)
                n = np.reshape(n, shape1)
                out = out.reshape(shape1)

                if self._method == "mean":
                    # For the average, the returned component is
                    # "sum", not "mean"
                    out = {"sum": out, "n": sum(counts)}
                else:
                    out = {self._method: out, "n": sum(counts)}
            else:
                # Return the reduced data as a numpy array. For most
                # methods the data is already in this form.
                if self._method == "mean":
                    # For the average, it is actually the sum that has
                    # been created, so we need to divide by the sample
                    # size.
                    out = out / sum(counts)

        return out

    def _process_chunk(self, fsref, chunk_coords, chunk_selection, out, counts,
                       out_selection, compressor, filters, missing, 
                       drop_axes=None):
        """
        Obtain part or whole of a chunk.

        This is done by taking binary data from storage and filling
        the output array.

        Note the need to use counts for some methods

        """
        coord = '.'.join([str(c) for c in chunk_coords])
        key = f"{self.ncvar}/{coord}"
        rfile, offset, size = tuple(fsref[key])

        if self.storage_type == "s3":
            USE_S3 = True
        else:
            USE_S3 = False
        if USE_S3:
            object = os.path.basename(rfile)
            tmp, count = s3_reduce_chunk(S3_ACTIVE_STORAGE_URL, S3_ACCESS_KEY,
                                         S3_SECRET_KEY, S3_URL, S3_BUCKET,
                                         object, offset, size,
                                         compressor, filters, missing,
                                         self.zds._dtype, self.zds._chunks,
                                         self.zds._order, chunk_selection,
                                         operation=self._method)
        else:
            # note there is an ongoing discussion about this interface, and what it returns
            # see https://github.com/valeriupredoi/PyActiveStorage/issues/33
            # so neither the returned data or the interface should be considered stable
            # although we will version changes.
            tmp, count = reduce_chunk(rfile, offset, size, compressor, filters,
                                      missing, self.zds._dtype,
                                      self.zds._chunks, self.zds._order,
                                      chunk_selection, method=self.method)

        if self.method is not None:
            out.append(tmp)
            counts.append(count)
        else:

            if drop_axes:
                tmp = np.squeeze(tmp, axis=drop_axes)

            # store selected data in output
            out[out_selection] = tmp
