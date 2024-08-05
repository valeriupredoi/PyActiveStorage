import pyfive
from activestorage.active import Active, get_missing_attributes
from activestorage.hdf2numcodec import decode_filters
import numpy as np

from activestorage import reductionist
from activestorage.active import load_from_s3
from activestorage.config import *
from test_bigger_data import save_cl_file_with_a

import json

class MockActive:
    def __init__(self, f, v):
        if USE_S3:
            self.f = load_from_s3(f)
        else:
            self.f = pyfive.File(f)
        ds = self.f[v]
        self.dtype = np.dtype(ds.dtype)
        self.array = pyfive.ZarrArrayStub(ds.shape, ds.chunks or ds.shape)
        self.missing = get_missing_attributes(ds)
        ds = ds._dataobjects
        self.ds = ds
    def __getitem__(self, args):
        if self.ds.filter_pipeline is None:
            compressor, filters = None, None
        else:
            compressor, filters = decode_filters(self.ds.filter_pipeline , self.dtype.itemsize, 'a')
        if self.ds.chunks is not None:
            self.ds._get_chunk_addresses()

        indexer = pyfive.OrthogonalIndexer(args, self.array)
        for chunk_coords, chunk_selection, out_selection in indexer:
            offset, size, filter_mask = self.ds.get_chunk_details(chunk_coords)
            jd = reductionist.build_request_data('a','b','c',
                                offset, size, compressor, filters, self.missing, self.dtype,
                                self.array._chunks,self.ds.order,chunk_selection)
            js = json.dumps(jd)
        return None

def test_build_request(tmp_path):

    ncfile, v = save_cl_file_with_a(tmp_path), 'cl'
    A = MockActive(ncfile,v)
    x = A[4:5, 1:2]
    # not interested in what is returned, checking that the request builds ok
