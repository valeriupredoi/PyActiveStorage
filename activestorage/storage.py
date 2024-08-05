"""Active storage module."""
import numpy as np

from numcodecs.compat import ensure_ndarray

def reduce_chunk(rfile, 
                 offset, size, compression, filters, missing, dtype, shape, 
                 order, chunk_selection, method=None):
    """ We do our own read of chunks and decoding etc 
    
    rfile - the actual file with the data 
    offset, size - where and what we want ...
    compression - optional `numcodecs.abc.Codec` compression codec
    filters - optional list of `numcodecs.abc.Codec` filter codecs
    dtype - likely float32 in most cases. 
    shape - will be a tuple, something like (3,3,1), this is the dimensionality of the 
            chunk itself
    order - typically 'C' for c-type ordering
    chunk_selection - python slice tuples for each dimension, e.g.
                        (slice(0, 2, 1), slice(1, 3, 1), slice(0, 1, 1))
                        this defines the part of the chunk which is to be obtained
                        or operated upon.
    method - computation desired 
            (in this Python version it's an actual method, in 
            storage implementations we'll change to controlled vocabulary)
                    
    """
    
    #FIXME: for the moment, open the file every time ... we might want to do that, or not
    with open(rfile,'rb') as open_file:
        # get the data
        chunk = read_block(open_file, offset, size)
        # reverse any compression and filters
        chunk = filter_pipeline(chunk, compression, filters)
        # make it a numpy array of bytes
        chunk = ensure_ndarray(chunk)
        # convert to the appropriate data type
        chunk = chunk.view(dtype)
        # sort out ordering and convert to the parent hyperslab dimensions
        chunk = chunk.reshape(-1, order='A')
        chunk = chunk.reshape(shape, order=order)

    tmp = chunk[chunk_selection]
    if method:
        if missing != (None, None, None, None):
            tmp = remove_missing(tmp, missing)
        # Check on size of tmp; method(empty) fails or gives incorrect
        # results
        if tmp.size:
            return method(tmp), tmp.size
        else:
            return tmp, 0
    else:
        return tmp, None


def filter_pipeline(chunk, compression, filters):
    """
    Reverse any compression and filters applied to the chunk.

    When a chunk is written, the filters are applied in order, then compression
    is applied. For reading, we must reverse this pipeline.

    :param chunk: possibly filtered and compressed bytes
    :param compression: optional `numcodecs.abc.Codec` compression codec
    :param filters: optional list of `numcodecs.abc.Codec` filter codecs
    :returns: decompressed and defiltered chunk bytes
    """
    if compression is not None:
        chunk = compression.decode(chunk)
    for filter in reversed(filters or []):
        chunk = filter.decode(chunk)
    return chunk


def remove_missing(data, missing):
    """ 
    As we are using numpy, we can use a masked array, storage implementations
    will have to do this by hand 
    """
    fill_value, missing_value, valid_min, valid_max = missing

    if fill_value:
        data = np.ma.masked_equal(data, fill_value)
    if missing_value:
        data = np.ma.masked_equal(data, missing_value)
    if valid_max:
        data = np.ma.masked_greater(data, valid_max)
    if valid_min:
        data = np.ma.masked_less(data, valid_min)

    data = np.ma.compressed(data)

    return data


def read_block(open_file, offset, size):
    """ Read <size> bytes from <open_file> at <offset>"""
    place = open_file.tell()
    open_file.seek(offset)
    data = open_file.read(size)
    open_file.seek(place)
    return data


def reduce_opens3_chunk(fh, 
        offset, size, compression, filters, missing, dtype, shape, 
        order, chunk_selection, method=None):
    """ 
    Same function as reduce_chunk, but this mimics what is done
    deep in the bowels of H5py/pyfive. The reason for doing this is
    so we can get per chunk metrics
    """
    fh.seek(offset)
    chunk_buffer = fh.read(size)
    chunk = filter_pipeline(chunk_buffer, compression, filters)
    # make it a numpy array of bytes
    chunk = ensure_ndarray(chunk)
    # convert to the appropriate data type
    chunk = chunk.view(dtype)
    # sort out ordering and convert to the parent hyperslab dimensions
    chunk = chunk.reshape(-1, order='A')
    chunk = chunk.reshape(shape, order=order)

    tmp = chunk[chunk_selection]
    if method:
        if missing != (None, None, None, None):
            tmp = remove_missing(tmp, missing)
        # check on size of tmp; method(empty) returns nan
        if tmp.any():
            return method(tmp), tmp.size
        else:
            return tmp, None
    else:
        return tmp, None

