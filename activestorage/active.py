import concurrent.futures
import os
import numpy as np
import pathlib
import urllib
import pyfive
import time

import s3fs

from activestorage.config import *
from activestorage import reductionist
from activestorage.storage import reduce_chunk, reduce_opens3_chunk
from activestorage.hdf2numcodec import decode_filters


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
   
    t1=time.time()   
    s3file = fs.open(uri, 'rb')
    t2=time.time()
    ds = pyfive.File(s3file)
    t3=time.time()
    print(f"Dataset loaded from S3 with s3fs and Pyfive: {uri} ({t2-t1:.2},{t3-t2:.2})")
    return ds

def _metricise(method):
    """ Decorator for class methods loads into metric_data"""
    def timed(self, *args, **kw):
        ts = time.time()
        metric_name=''
        if '__metric_name' in kw:
            metric_name = kw['__metric_name']
            del kw['__metric_name']
        result = method(self,*args, **kw)
        te = time.time()
        if metric_name:
            self.metric_data[metric_name] = te-ts
        return result
    return timed


def get_missing_attributes(ds):
    """" 
    Load all the missing attributes we need from a netcdf file
    """

    def hfix(x):
        '''
        return item if single element list/array
        see https://github.com/h5netcdf/h5netcdf/issues/116
        '''
        if x is None:
            return x
        if not np.isscalar(x) and len(x) == 1:
            return x[0]
        return x

    _FillValue = hfix(ds.attrs.get('_FillValue'))
    missing_value = ds.attrs.get('missing_value')
    valid_min = hfix(ds.attrs.get('valid_min'))
    valid_max = hfix(ds.attrs.get('valid_max'))
    valid_range = hfix(ds.attrs.get('valid_range'))
    if valid_max is not None or valid_min is not None:
        if valid_range is not None:
            raise ValueError(
                "Invalid combination in the file of valid_min, "
                "valid_max, valid_range: "
                f"{valid_min}, {valid_max}, {valid_range}"
            )
    elif valid_range is not None:            
        valid_min, valid_max = valid_range
    
    return _FillValue, missing_value, valid_min, valid_max

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
        max_threads=100,
        storage_options=None,
        active_storage_url=None
    ):
        """
        Instantiate with a NetCDF4 dataset URI and the variable of interest within that file.
        (We need the variable, because we need variable specific metadata from within that
        file, however, if that information is available at instantiation, it can be provided
        using keywords and avoid a metadata read.)

        :param storage_options: s3fs.S3FileSystem options
        :param active_storage_url: Reductionist server URL
        """
        # Assume NetCDF4 for now
        self.uri = uri
        if self.uri is None:
            raise ValueError(f"Must use a valid file for uri. Got {uri}")

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

        self._version = 1
        self._components = False
        self._method = None
        self._max_threads = max_threads
        self.missing = None
        self.ds = None
        self.metric_data = {}
        self.data_read = 0

    @_metricise
    def __load_nc_file(self):
        """ Get the netcdf file and it's b-tree"""
        ncvar = self.ncvar
        # in all cases we need an open netcdf file to get at attributes
        # we keep it open because we need it's b-tree
        if self.storage_type is None:
            nc = pyfive.File(self.uri)
        elif self.storage_type == "s3":
            nc = load_from_s3(self.uri, self.storage_options)
        self.filename = self.uri

        self.ds = nc[ncvar]

    def __get_missing_attributes(self):
        if self.ds is None:
            self.__load_nc_file()
        return get_missing_attributes(self.ds)

    def __getitem__(self, index):
        """ 
        Provides support for a standard get item.
        #FIXME-BNL: Why is the argument index?
        """
        self.metric_data = {}
        if self.ds is None:
            self.__load_nc_file(__metric_name='load nc time')
            #self.__metricise('Load','__load_nc_file')
        
        self.missing = self.__get_missing_attributes()

        self.data_read = 0
        
        if self.method is None and self._version == 0:
        
            # No active operation
            return self._get_vanilla(index, __metric_name='vanilla_time')

        elif self._version == 1:

            #FIXME: is the difference between version 1 and 2 still honoured?
            return self._get_selection(index, __metric_name='selection 1 time (s)')
        
        elif self._version  == 2:

            return self._get_selection(index, __metric_name='selection 2 time (s)')
          
        else:
            raise ValueError(f'Version {self._version} not supported')

    @_metricise
    def _get_vanilla(self, index):
        """ 
        Get the data without any active operation
        """
        data = self.ds[index]    
        data = self._mask_data(data)
        return data

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


    def _get_active(self, method, *args):
        """ 
        *args defines a slice of data. This method loops over each of the chunks
        necessary to extract the parts of the slice, and asks the active storage 
        to apply the method to each part. It then applies the method to 
        the partial results and returns a value is if  method had been applied to
        an array returned via getitem.
        """
        raise NotImplementedError
 
    @_metricise
    def _get_selection(self, *args):
        """ 
        At this point we have a Dataset object, but all the important information about
        how to use it is in the attribute DataoobjectDataset class. Here we gather 
        metadata from the dataset instance and then continue with the dataobjects instance.
        """

        # stick this here for later, to discuss with David
        keepdims = True

        name = self.ds.name
        dtype = np.dtype(self.ds.dtype)
        # hopefully fix pyfive to get a dtype directly
        array = pyfive.ZarrArrayStub(self.ds.shape, self.ds.chunks)
        ds = self.ds._dataobjects
        
        self.metric_data['args'] = args
        self.metric_data['dataset shape'] = self.ds.shape
        self.metric_data['dataset chunks'] = self.ds.chunks
        if ds.filter_pipeline is None:
            compressor, filters = None, None
        else:
            compressor, filters = decode_filters(ds.filter_pipeline , dtype.itemsize, name)
        
        indexer = pyfive.OrthogonalIndexer(*args, array)
        out_shape = indexer.shape
        #stripped_indexer = [(a, b, c) for a,b,c in indexer]
        drop_axes = indexer.drop_axes and keepdims

        # we use array._chunks rather than ds.chunks, as the latter is none in the case of
        # unchunked data, and we need to tell the storage the array dimensions in this case.
        return self._from_storage(ds, indexer, array._chunks, out_shape, dtype, compressor, filters, drop_axes)

    def _from_storage(self, ds, indexer, chunks, out_shape, out_dtype, compressor, filters, drop_axes):
        method = self.method
       
        if method is not None:
            out = []
            counts = []
        else:
            out = np.empty(out_shape, dtype=out_dtype, order=ds.order)
            counts = None  # should never get touched with no method!

        # Create a shared session object.
        if self.storage_type == "s3" and self._version==2:
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
        # Because we do this, we need to read the dataset b-tree now, not as we go, so
        # it is already in cache. If we remove the thread pool from here, we probably
        # wouldn't need to do it before the first one.
        
        if ds.chunks is not None:
            t1 = time.time()
            ds._get_chunk_addresses()
            t2 = time.time() - t1
            self.metric_data['indexing time (s)'] = t2
            self.metric_data['chunk number'] = len(ds._zchunk_index)
        chunk_count = 0
        t1 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_threads) as executor:
            futures = []
            # Submit chunks for processing.
            for chunk_coords, chunk_selection, out_selection in indexer:
                future = executor.submit(
                    self._process_chunk,
                    session,  ds, chunks, chunk_coords, chunk_selection,
                    counts, out_selection, compressor, filters, drop_axes=drop_axes)
                futures.append(future)
            # Wait for completion.
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    raise
                else:
                    chunk_count +=1
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

        t2 = time.time()
        self.metric_data['reduction time (s)'] = t2-t1
        self.metric_data['chunks processed'] = chunk_count
        self.metric_data['storage read (B)'] = self.data_read
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

    def _process_chunk(self, session, ds, chunks, chunk_coords, chunk_selection, counts,
                       out_selection, compressor, filters, drop_axes=None):
        """
        Obtain part or whole of a chunk.

        This is done by taking binary data from storage and filling
        the output array.

        Note the need to use counts for some methods
        #FIXME: Do, we, it's not actually used?

        """
        
        offset, size, filter_mask = ds.get_chunk_details(chunk_coords)
        self.data_read += size

        if self.storage_type == 's3' and self._version == 1:

            tmp, count = reduce_opens3_chunk(ds.fh, offset, size, compressor, filters,
                            self.missing, ds.dtype,
                            chunks, ds.order,
                            chunk_selection, method=self.method
            )

        elif self.storage_type == "s3" and self._version==2:
            # S3: pass in pre-configured storage options (credentials)
            # print("S3 rfile is:", self.filename)
            parsed_url = urllib.parse.urlparse(self.filename)
            bucket = parsed_url.netloc
            object = parsed_url.path
        
            # for certain S3 servers rfile needs to contain the bucket eg "bucket/filename"
            # as a result the parser above finds empty string bucket
            if bucket == "":
                bucket = os.path.dirname(object)
                object = os.path.basename(object)
            # print("S3 bucket:", bucket)
            # print("S3 file:", object)
            if self.storage_options is None:
                # for the moment we need to force ds.dtype to be a numpy type
                tmp, count = reductionist.reduce_chunk(session,
                                                       S3_ACTIVE_STORAGE_URL,
                                                       S3_URL,
                                                       bucket, object, offset,
                                                       size, compressor, filters,
                                                       self.missing, np.dtype(ds.dtype),
                                                       chunks,
                                                       ds.order,
                                                       chunk_selection,
                                                       operation=self._method)
            else:
                # special case for "anon=True" buckets that work only with e.g.
                # fs = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': S3_URL})
                # where file uri = bucketX/fileY.mc
                print("S3 Storage options to Reductionist:", self.storage_options)
                if self.storage_options.get("anon", None) == True:
                    bucket = os.path.dirname(parsed_url.path)  # bucketX
                    object = os.path.basename(parsed_url.path)  # fileY
                    print("S3 anon=True Bucket and File:", bucket, object)
                tmp, count = reductionist.reduce_chunk(session,
                                                       self.active_storage_url,
                                                       self._get_endpoint_url(),
                                                       bucket, object, offset,
                                                       size, compressor, filters,
                                                       self.missing, np.dtype(ds.dtype),
                                                       chunks,
                                                       ds.order,
                                                       chunk_selection,
                                                       operation=self._method)
        elif self.storage_type=='ActivePosix' and self.version==2:
            # This is where the DDN Fuse and Infinia wrappers go
            raise NotImplementedError
        else:
            # note there is an ongoing discussion about this interface, and what it returns
            # see https://github.com/valeriupredoi/PyActiveStorage/issues/33
            # so neither the returned data or the interface should be considered stable
            # although we will version changes.
            tmp, count = reduce_chunk(self.filename, offset, size, compressor, filters,
                                      self.missing, ds.dtype,
                                      chunks, ds.order,
                                      chunk_selection, method=self.method)

        if self.method is not None:
            return tmp, count 
        else:
            if drop_axes:
                tmp = np.squeeze(tmp, axis=drop_axes)
            return tmp, out_selection 

    def _mask_data(self, data):
        """ 
        Missing values obtained at initial getitem, and are used here to 
        mask data, if necessary
        """
        if self.missing is None:
            self.missing = self.__get_missing_attributes()
        _FillValue, missing_value, valid_min, valid_max = self.missing
        
        if _FillValue is not None:
            data = np.ma.masked_equal(data, _FillValue)

        if missing_value is not None:
            data = np.ma.masked_equal(data, missing_value)

        if valid_max is not None:
            data = np.ma.masked_greater(data, valid_max)

        if valid_min is not None:
            data = np.ma.masked_less(data, valid_min)

        return data
