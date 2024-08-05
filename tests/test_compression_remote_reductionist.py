import os
import numpy as np
import pytest

import utils

from netCDF4 import Dataset
from pathlib import Path

from activestorage.active import Active, load_from_s3
from activestorage.config import *
from activestorage.dummy_data import make_compressed_ncdata
from activestorage.reductionist import ReductionistError as RedErr


# Bryan's S3 machine + Bryan's reductionist
STORAGE_OPTIONS_Bryan = {
    'anon': True,
    'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"},
}
S3_ACTIVE_URL_Bryan = "https://192.171.169.248:8080"
# TODO include all supported configuration types
storage_options_paramlist = [
    (STORAGE_OPTIONS_Bryan, S3_ACTIVE_URL_Bryan)
]
# bucket needed too for this test only
# otherwise, bucket is extracted automatically from full file uri
S3_BUCKET = "bnl"


# CMIP6_test.nc keeps being unavailable due to BNL bucket unavailable
@pytest.mark.xfail(reason='JASMIN messing about with SOF.')
@pytest.mark.parametrize("storage_options, active_storage_url", storage_options_paramlist)
def test_compression_and_filters_cmip6_data(storage_options, active_storage_url):
    """
    Test use of datasets with compression and filters applied for a real
    CMIP6 dataset (CMIP6-test.nc) - an IPSL file.

    This test will always pass when USE_S3 = False; equally, it will always
    fail if USE_S3 = True until Reductionist supports anon=True S3 buckets.
    See following test below with a forced storage_type="s3" that mimicks
    locally the fail, and catches it. Equally, we catch the same exception when USE_S3=True

    Important info on session data:
    S3 Storage options to Reductionist: {'anon': True, 'client_kwargs': {'endpoint_url': 'https://uor-aces-o.s3-ext.jc.rl.ac.uk'}}
    S3 anon=True Bucket and File: bnl CMIP6-test.nc
    Reductionist request data dictionary: {'source': 'https://uor-aces-o.s3-ext.jc.rl.ac.uk', 'bucket': 'bnl', 'object': 'CMIP6-test.nc', 'dtype': 'float32', 'byte_order': 'little', 'offset': 29385, 'size': 942518, 'order': 'C', 'shape': (15, 143, 144), 'selection': [[0, 2, 1], [4, 6, 1], [7, 9, 1]], 'compression': {'id': 'zlib'}}
    """
    test_file = str(Path(__file__).resolve().parent / 'test_data' / 'CMIP6-test.nc')
    with Dataset(test_file) as nc_data:
        nc_min = np.min(nc_data["tas"][0:2,4:6,7:9])
    print(f"Numpy min from compressed file {nc_min}")

    # TODO remember that the special case for "anon=True" buckets is that
    # the actual file uri = "bucket/filename"
    if USE_S3:
        ofile = os.path.basename(test_file)
        test_file_uri = os.path.join(S3_BUCKET, ofile)
    else:
        test_file_uri = test_file
    active = Active(test_file_uri, 'tas', utils.get_storage_type(),
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)
    active._version = 1
    active._method = "min"

    if USE_S3:
        # for now anon=True S3 buckets are not supported by Reductionist
        with pytest.raises(RedErr) as rederr:
            result = active[0:2,4:6,7:9]
        access_denied_err = 'code: \\"AccessDenied\\"'
        assert access_denied_err in str(rederr.value)
        # assert nc_min == result
        # assert result == 239.25946044921875
    else:
        result = active[0:2,4:6,7:9]
        assert nc_min == result
        assert result == 239.25946044921875


# CMIP6_test.nc keeps being unavailable due to BNL bucket unavailable
@pytest.mark.xfail(reason='JASMIN messing about with SOF.')
@pytest.mark.parametrize("storage_options, active_storage_url", storage_options_paramlist)
def test_compression_and_filters_cmip6_forced_s3_from_local(storage_options, active_storage_url):
    """
    Test use of datasets with compression and filters applied for a real
    CMIP6 dataset (CMIP6-test.nc) - an IPSL file.

    This is for a special anon=True bucket ONLY.
    """
    test_file = str(Path(__file__).resolve().parent / 'test_data' / 'CMIP6-test.nc')
    with Dataset(test_file) as nc_data:
        nc_min = np.min(nc_data["tas"][0:2,4:6,7:9])
    print(f"Numpy min from compressed file {nc_min}")

    # TODO remember that the special case for "anon=True" buckets is that
    # the actual file uri = "bucket/filename"
    ofile = os.path.basename(test_file)
    test_file_uri = os.path.join(S3_BUCKET, ofile)
    active = Active(test_file_uri, 'tas', storage_type="s3",
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)

    active._version = 1
    active._method = "min"

    # for now anon=True S3 buckets are not supported by Reductionist
    with pytest.raises(RedErr) as rederr:
        result = active[0:2,4:6,7:9]
    access_denied_err = 'code: \\"AccessDenied\\"'
    assert access_denied_err in str(rederr.value)
    # assert nc_min == result
    # assert result == 239.25946044921875


# CMIP6_test.nc keeps being unavailable due to BNL bucket unavailable
@pytest.mark.xfail(reason='JASMIN messing about with SOF.')
def test_compression_and_filters_cmip6_forced_s3_from_local_2():
    """
    Test use of datasets with compression and filters applied for a real
    CMIP6 dataset (CMIP6-test.nc) - an IPSL file.

    This is for a special anon=True bucket connected to via valid key.secret
    """
    storage_options = {
        'key': "f2d55c6dcfc7618b2c34e00b58df3cef",
        'secret': "$/'#M{0{/4rVhp%n^(XeX$q@y#&(NM3W1->~N.Q6VP.5[@bLpi='nt]AfH)>78pT",
        'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"}
    }
    active_storage_url = "https://192.171.169.248:8080"
    test_file = str(Path(__file__).resolve().parent / 'test_data' / 'CMIP6-test.nc')
    with Dataset(test_file) as nc_data:
        nc_min = np.min(nc_data["tas"][0:2,4:6,7:9])
    print(f"Numpy min from compressed file {nc_min}")

    ofile = os.path.basename(test_file)
    test_file_uri = os.path.join(
        S3_BUCKET,
        ofile
    )
    print("S3 Test file path:", test_file_uri)
    active = Active(test_file_uri, 'tas', storage_type="s3",
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)

    active._version = 1
    active._method = "min"

    result = active[0:2,4:6,7:9]
    assert nc_min == result
    assert result == 239.25946044921875


# CMIP6_test.nc keeps being unavailable due to BNL bucket unavailable
@pytest.mark.xfail(reason='JASMIN messing about with SOF.')
@pytest.mark.skipif(not USE_S3, reason="we need only localhost Reductionist in GA CI")
@pytest.mark.skipif(REMOTE_RED, reason="we need only localhost Reductionist in GA CI")
def test_compression_and_filters_cmip6_forced_s3_using_local_Reductionist():
    """
    Test use of datasets with compression and filters applied for a real
    CMIP6 dataset (CMIP6-test.nc) - an IPSL file.

    This is for a special anon=True bucket connected to via valid key.secret
    and uses the locally deployed Reductionist via container.
    """
    print("Reductionist URL", S3_ACTIVE_STORAGE_URL)
    storage_options = {
        'key': "f2d55c6dcfc7618b2c34e00b58df3cef",
        'secret': "$/'#M{0{/4rVhp%n^(XeX$q@y#&(NM3W1->~N.Q6VP.5[@bLpi='nt]AfH)>78pT",
        'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"}
    }

    test_file = str(Path(__file__).resolve().parent / 'test_data' / 'CMIP6-test.nc')
    with Dataset(test_file) as nc_data:
        nc_min = np.min(nc_data["tas"][0:2,4:6,7:9])
    print(f"Numpy min from compressed file {nc_min}")

    ofile = os.path.basename(test_file)
    test_file_uri = os.path.join(
        S3_BUCKET,
        ofile
    )
    print("S3 Test file path:", test_file_uri)
    active = Active(test_file_uri, 'tas', storage_type="s3",
                    storage_options=storage_options,
                    active_storage_url=S3_ACTIVE_STORAGE_URL)

    active._version = 1
    active._method = "min"

    result = active[0:2,4:6,7:9]
    assert nc_min == result
    assert result == 239.25946044921875


# if pytest is run on a single thread, this test throws a PytestUnraisableException
# followed at the end by a SegFault (test passes fine, and writes report etc); when
# pytest runs in n>=2 threads all is fine. This is defo due to something in Kerchunk
# tests/test_compression_remote_reductionist.py::test_compression_and_filters_cmip6_forced_s3_from_local_bigger_file_v1
#   /home/valeriu/miniconda3/envs/pyactive/lib/python3.12/site-packages/_pytest/unraisableexception.py:80: PytestUnraisableExceptionWarning: Exception ignored in: 'h5py._objects.ObjectID.__dealloc__'
#   
#   Traceback (most recent call last):
#     File "h5py/_objects.pyx", line 201, in h5py._objects.ObjectID.__dealloc__
#     File "h5py/h5fd.pyx", line 180, in h5py.h5fd.H5FD_fileobj_truncate
#   AttributeError: 'S3File' object has no attribute 'truncate'
# For no I am just shutting this up, later we may have to take it up with Kerchunk
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_compression_and_filters_cmip6_forced_s3_from_local_bigger_file_v1():
    """
    Test identical to previous
    test_compression_and_filters_cmip6_forced_s3_from_local_2
    but using a bigger file, for more relevant performance measures.

    This is for a special anon=True bucket connected to via valid key.secret
    Variable standard_name: tendency_of_eastward_wind_due_to_orographic_gravity_wave_drag
    Variable var_name: m01s06i247_4
    dims: 30 (time) 39 (plev) 325 (lat) 432 (lon)

    Entire mother file info:
    mc ls bryan/bnl/da193a_25_day__198808-198808.nc
    [2024-01-24 10:07:03 GMT] 2.8GiB STANDARD da193a_25_day__198808-198808.nc
    

    NOTE: we used this test as timing reference for performance testing.
    """
    storage_options = {
        'key': "f2d55c6dcfc7618b2c34e00b58df3cef",
        'secret': "$/'#M{0{/4rVhp%n^(XeX$q@y#&(NM3W1->~N.Q6VP.5[@bLpi='nt]AfH)>78pT",
        'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"}
    }
    active_storage_url = "https://192.171.169.248:8080"
    bigger_file = "da193a_25_day__198808-198808.nc"

    test_file_uri = os.path.join(
        S3_BUCKET,
        bigger_file
    )
    print("S3 Test file path:", test_file_uri)
    active = Active(test_file_uri, 'm01s06i247_4', storage_type="s3",
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)

    active._version = 1
    active._method = "min"

    result = active[0:2, 0:3, 4:6, 7:9]


def test_compression_and_filters_cmip6_forced_s3_from_local_bigger_file_v0():
    """
    Test identical to previous
    test_compression_and_filters_cmip6_forced_s3_from_local_2
    but using a bigger file, for more relevant performance measures.

    This is for a special anon=True bucket connected to via valid key.secret
    Variable standard_name: tendency_of_eastward_wind_due_to_orographic_gravity_wave_drag
    Variable var_name: m01s06i247_4
    dims: 30 (time) 39 (plev) 325 (lat) 432 (lon)

    Entire mother file info:
    mc ls bryan/bnl/da193a_25_day__198808-198808.nc
    [2024-01-24 10:07:03 GMT] 2.8GiB STANDARD da193a_25_day__198808-198808.nc
    
    """
    storage_options = {
        'key': "f2d55c6dcfc7618b2c34e00b58df3cef",
        'secret': "$/'#M{0{/4rVhp%n^(XeX$q@y#&(NM3W1->~N.Q6VP.5[@bLpi='nt]AfH)>78pT",
        'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"}
    }
    active_storage_url = "https://192.171.169.248:8080"
    bigger_file = "da193a_25_day__198808-198808.nc"

    test_file_uri = os.path.join(
        S3_BUCKET,
        bigger_file
    )
    print("S3 Test file path:", test_file_uri)
    active = Active(test_file_uri, 'm01s06i247_4', storage_type="s3",
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)

    active._version = 0
    d = active[0:2, 0:3, 4:6, 7:9]
    min_result = np.min(d)
    print(min_result)
