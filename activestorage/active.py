import concurrent.futures
import contextlib
import os
import numpy as np
import pathlib
import urllib

import h5netcdf
import s3fs

#FIXME: Consider using h5py throughout, for more generality
from netCDF4 import Dataset
from zarr.indexing import (
    OrthogonalIndexer,
)
from activestorage.config import *
from activestorage import reductionist
from activestorage.storage import reduce_chunk
from activestorage import netcdf_to_zarr as nz

import time
import zarr

t1 = time.time()
@contextlib.contextmanager
def load_from_s3(uri, storage_options=None):
    """
    Load a netCDF4-like object from S3.

    First, set up an S3 filesystem with s3fs.S3FileSystem.
    Then open the uri with this FS -> s3file
    s3file is a File-like object: a memory view but wih all the metadata
    gubbins inside it (no data!)
    calling >> ds = netCDF4.Dataset(s3file) <<
    will throw a FileNotFoundError because the netCDF4 library is always looking for
    a local file, resulting in [Errno 2] No such file or directory:
    '<File-like object S3FileSystem, pyactivestorage/s3_test_bizarre.nc>'
    instead, we use h5netcdf: https://github.com/h5netcdf/h5netcdf
    a Python binder straight to HDF5-netCDF4 interface, that doesn't need a "local" file

    storage_options: kwarg dict containing S3 credentials passed straight to Active
    """
    if storage_options is None:  # use pre-configured S3 credentials
        fs = s3fs.S3FileSystem(key=S3_ACCESS_KEY,  # eg "minioadmin" for Minio
                               secret=S3_SECRET_KEY,  # eg "minioadmin" for Minio
                               client_kwargs={'endpoint_url': S3_URL})  # eg "http://localhost:9000" for Minio
    else:
        fs = s3fs.S3FileSystem(**storage_options)  # use passed-in dictionary
    with fs.open(uri, 'rb') as s3file:
        ds = h5netcdf.File(s3file, 'r', invalid_netcdf=True)
        print(f"Dataset loaded from S3 via h5netcdf: {ds}")
        yield ds


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

    def __init__(
        self,
        uri,
        ncvar,
        storage_type=None,
        max_threads=150,
        storage_options=None,
        active_storage_url=None
    ):
        """
        Instantiate with a NetCDF4 dataset and the variable of interest within that file.
        (We need the variable, because we need variable specific metadata from within that
        file, however, if that information is available at instantiation, it can be provided
        using keywords and avoid a metadata read.)

        :param storage_options: s3fs.S3FileSystem options
        :param active_storage_url: Reductionist server URL
        """
        # Assume NetCDF4 for now
        self.uri = uri
        if self.uri is None:
            raise ValueError(f"Must use a valid file for uri. Got {self.uri}")

        # still allow for a passable storage_type
        # for special cases eg "special-POSIX" ie DDN
        if not storage_type and storage_options is not None:
            storage_type = urllib.parse.urlparse(uri).scheme
        self.storage_type = storage_type

        # get storage_options
        self.storage_options = storage_options
        self.active_storage_url = active_storage_url

        # basic check on file
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
        self._max_threads = max_threads

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

            if self.storage_type is None:
                nc = Dataset(self.uri)
                data = nc[ncvar][index]
                nc.close()
            elif self.storage_type == "s3":
                with load_from_s3(self.uri, self.storage_options) as nc:
                    data = nc[ncvar][index]
                    data = self._mask_data(data, nc[ncvar])

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
        tx = time.time()
        # FIXME: Order of calls is hardcoded'
        if self.zds is None:
            print(f"Kerchunking file {self.uri} with variable "
                  f"{self.ncvar} for storage type {self.storage_type}")
            tx1 = time.time()
            ds, zarray, zattrs = nz.load_netcdf_zarr_generic(
                self.uri,
                self.ncvar,
                self.storage_type,
                self.storage_options,
            )
            ty1 = time.time()
            print("Time to load to netCDF from Zarr", ty1 - tx1)
            # The following is a hangove from exploration
            # and is needed if using the original doing it ourselves
            # self.zds = make_an_array_instance_active(ds)
            if isinstance(ds, zarr.hierarchy.Group):
                ds = ds[self.ncvar]
            self.zds = ds

            # Retain attributes and other information
            if zarray.get('fill_value') is not None:
                zattrs['_FillValue'] = zarray['fill_value']
            
            self.zarray = zarray
            self.zattrs = zattrs

            # FIXME: We do not get the correct byte order on the Zarr
            # Array's dtype when using S3, so capture it here.
            self._dtype = np.dtype(zarray['dtype'])

        ty = time.time()
        print("Time of _via_kerchunk()", ty - tx)
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

        # Get missing values
        _FillValue = self.zattrs.get('_FillValue')
        missing_value = self.zattrs.get('missing_value')
        valid_min = self.zattrs.get('valid_min')
        valid_max = self.zattrs.get('valid_max')
        valid_range = self.zattrs.get('valid_range')
        if valid_max is not None or valid_min is not None:
            if valid_range is not None:
                raise ValueError(
                    "Invalid combination in the file of valid_min, "
                    "valid_max, valid_range: "
                    f"{valid_min}, {valid_max}, {valid_range}"
                )
        elif valid_range is not None:            
            valid_min, valid_max = valid_range
        
        missing = (
            _FillValue,
            missing_value,
            valid_min,
            valid_max,
        )

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

        # Create a shared session object.
        if self.storage_type == "s3":
            if self.storage_options is not None:
                key, secret = None, None
                if "key" in self.storage_options:
                    key = self.storage_options["key"]
                if "secret" in self.storage_options:
                    secret = self.storage_options["secret"]
                if key and secret:
                    session = reductionist.get_session(key, secret,
                                                       S3_ACTIVE_STORAGE_CACERT)
                else:
                    session = reductionist.get_session(S3_ACCESS_KEY, S3_SECRET_KEY,
                                                       S3_ACTIVE_STORAGE_CACERT)
            else:
                session = reductionist.get_session(S3_ACCESS_KEY, S3_SECRET_KEY,
                                                   S3_ACTIVE_STORAGE_CACERT)
        else:
            session = None

        # Process storage chunks using a thread pool.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_threads) as executor:
            futures = []
            # Submit chunks for processing.
            for chunk_coords, chunk_selection, out_selection in stripped_indexer:
                future = executor.submit(
                    self._process_chunk,
                    session, fsref, chunk_coords, chunk_selection,
                    counts, out_selection,
                    compressor, filters, missing,
                    drop_axes=drop_axes)
                futures.append(future)
            # Wait for completion.
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    raise
                else:
                    if method is not None:
                        result, count = result
                        out.append(result)
                        counts.append(count)
                    else:
                        # store selected data in output
                        result, selection = result
                        out[selection] = result

        if method is not None:
            # Apply the method (again) to aggregate the result
            out = method(out)
            shape1 = (1,) * len(out_shape)
                
            if self._components:
                # Return a dictionary of components containing the
                # reduced data and the sample size ('n'). (Rationale:
                # cf-python needs the sample size for all reductions;
                # see the 'mtol' parameter of cf.Field.collapse.)
                #
                # Note that in all components must always have the
                # same number of dimensions as the original array,
                # i.e. 'drop_axes' is always considered False,
                # regardless of its setting. (Rationale: dask
                # reductions require the per-dask-chunk partial
                # reductions to retain these dimensions so that
                # partial results can be concatenated correctly.)
                out = out.reshape(shape1)

                n = np.sum(counts).reshape(shape1)
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
                    out = out / np.sum(counts).reshape(shape1)

        return out

    def _get_endpoint_url(self):
        """Return the endpoint_url of an S3 object store, or `None`"""
        endpoint_url = self.storage_options.get('endpoint_url')
        if endpoint_url is not None:
            return endpoint_url

        client_kwargs = self.storage_options.get('client_kwargs')
        if client_kwargs:
            endpoint_url = client_kwargs.get('endpoint_url')
            if endpoint_url is not None:
                return endpoint_url

        return f"http://{urllib.parse.urlparse(self.filename).netloc}"

    def _process_chunk(self, session, fsref, chunk_coords, chunk_selection, counts,
                       out_selection, compressor, filters, missing, 
                       drop_axes=None):
        """
        Obtain part or whole of a chunk.

        This is done by taking binary data from storage and filling
        the output array.

        Note the need to use counts for some methods

        """
        t1c = time.time()
        coord = '.'.join([str(c) for c in chunk_coords])
        key = f"{self.ncvar}/{coord}"
        try:
            rfile, offset, size = tuple(fsref[self.ncvar + " /" + key])
        except KeyError:
            rfile, offset, size = tuple(fsref[key])

        # S3: pass in pre-configured storage options (credentials)
        if self.storage_type == "s3":
            # print("S3 rfile is:", rfile)
            parsed_url = urllib.parse.urlparse(rfile)
            bucket = parsed_url.netloc
            object = parsed_url.path
            # FIXME: We do not get the correct byte order on the Zarr Array's dtype
            # when using S3, so use the value captured earlier.
            dtype = self._dtype
            # for certain S3 servers rfile needs to contain the bucket eg "bucket/filename"
            # as a result the parser above finds empty string bucket
            if bucket == "":
                bucket = os.path.dirname(object)
                object = os.path.basename(object)
            # print("S3 bucket:", bucket)
            # print("S3 file:", object)
            if self.storage_options is None:
                tmp, count = reductionist.reduce_chunk(session,
                                                       S3_ACTIVE_STORAGE_URL,
                                                       S3_URL,
                                                       bucket, object, offset,
                                                       size, compressor, filters,
                                                       missing, dtype,
                                                       self.zds._chunks,
                                                       self.zds._order,
                                                       chunk_selection,
                                                       operation=self._method)
            else:
                # special case for "anon=True" buckets that work only with e.g.
                # "fs = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': S3_URL})"
                # where file uri = bucketX/fileY.mc
                # print("S3 Storage options:", self.storage_options)
                if self.storage_options.get("anon", None) == True:
                    bucket = os.path.dirname(parsed_url.path)  # bucketX
                    object = os.path.basename(parsed_url.path)  # fileY
                    print("S3 anon=True Bucket and File:", bucket, object)
                tmp, count = reductionist.reduce_chunk(session,
                                                       self.active_storage_url,
                                                       self._get_endpoint_url(),
                                                       bucket, object, offset,
                                                       size, compressor, filters,
                                                       missing, dtype,
                                                       self.zds._chunks,
                                                       self.zds._order,
                                                       chunk_selection,
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
        t2c = time.time()
        print("Chunk processing time via _process_chunk()", t2c - t1c)
        if self.method is not None:
            return tmp, count
        else:
            if drop_axes:
                tmp = np.squeeze(tmp, axis=drop_axes)
            return tmp, out_selection

    def _mask_data(self, data, ds_var):
        """ppp"""
        # TODO: replace with cfdm.NetCDFIndexer, hopefully.
        attrs = ds_var.attrs
        missing_value = attrs.get('missing_value')
        _FillValue = attrs.get('_FillValue')
        valid_min = attrs.get('valid_min')
        valid_max = attrs.get('valid_max')
        valid_range = attrs.get('valid_range')

        if valid_max is not None or valid_min is not None:
            if valid_range is not None:
                raise ValueError(
                    "Invalid combination in the file of valid_min, "
                    "valid_max, valid_range: "
                    f"{valid_min}, {valid_max}, {valid_range}"
                )
        elif valid_range is not None:
            valid_min, valid_max = valid_range
        
        if _FillValue is not None:
            data = np.ma.masked_equal(data, _FillValue)

        if missing_value is not None:
            data = np.ma.masked_equal(data, missing_value)

        if valid_max is not None:
            data = np.ma.masked_greater(data, valid_max)

        if valid_min is not None:
            data = np.ma.masked_less(data, valid_min)

        return data
