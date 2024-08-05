import numcodecs

def decode_filters(filter_pipeline, itemsize, name):
    """
    
    Convert HDF5 filter and compression instructions into instructions understood
    by numcodecs. Input is a pyfive filter_pipeline object, the itemsize, and the
    dataset name for error messages.

    We can only support things that are supported by our storage backends, which
    is a rather limited range right now:
     - gzip compression, and
     - shuffle filter 

    See aslso zarr kerchunk _decode_filters. We may need to add much more 
    support for BLOSC than currently supported.

    Useful notes on filters are here:
     - https://docs.unidata.ucar.edu/netcdf-c/current/filters.html and
     - https://docs.hdfgroup.org/hdf5/v1_8/group___h5_z.html

    In particular, note that we can only support things that go beyond native HDF5
    if _we_ support them directly.

    """
    compressors, filters = [], []

    for filter in filter_pipeline:

        filter_id=filter['filter_id']
        properties = filter['client_data_values']


        # We suppor the following
        if filter_id == GZIP_DEFLATE_FILTER:
            compressors.append(numcodecs.Zlib(level=properties[0]))
        elif filter_id == SHUFFLE_FILTER:
            filters.append(numcodecs.Shuffle(elementsize=itemsize))
        else:
            raise NotImplementedError('We cannot yet support filter id ',filter_id)
        
        # We might be able, in the future, to support the following
        # At the moment the following code cannot be implemented, but we can move 
        # the loops up as we develop backend support.

        if 0:


            if filter_id == 32001:
                blosc_compressors = (
                    "blosclz",
                    "lz4",
                    "lz4hc",
                    "snappy",
                    "zlib",
                    "zstd",
                )
                (
                    _1,
                    _2,
                    bytes_per_num,
                    total_bytes,
                    clevel,
                    shuffle,
                    compressor,
                ) = properties
                pars = dict(
                    blocksize=total_bytes,
                    clevel=clevel,
                    shuffle=shuffle,
                    cname=blosc_compressors[compressor],
                )
                filters.append(numcodecs.Blosc(**pars))
            elif filter_id == 32015:
                filters.append(numcodecs.Zstd(level=properties[0]))
            elif filter_id == 32004:
                raise RuntimeError(
                    f"{name} uses lz4 compression - not supported as yet"
                )
            elif filter_id == 32008:
                raise RuntimeError(
                    f"{name} uses bitshuffle compression - not supported as yet"
                )
            else:
                raise RuntimeError(
                    f"{name} uses filter id {filter_id} with properties {properties},"
                    f" not supported as yet"
                )
            
    if len(compressors) > 1: 
        raise ValueError('We only expected one compression algorithm')
    return compressors[0], filters

# These are from pyfive's HDF5 filter definitions
# IV.A.2.l The Data Storage - Filter Pipeline message
RESERVED_FILTER = 0
GZIP_DEFLATE_FILTER = 1
SHUFFLE_FILTER = 2
FLETCH32_FILTER = 3
SZIP_FILTER = 4
NBIT_FILTER = 5
SCALEOFFSET_FILTER = 6
