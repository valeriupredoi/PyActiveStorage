import os
import numpy as np
import pytest

from netCDF4 import Dataset
from pathlib import Path

from activestorage.active import Active, load_from_s3
from activestorage.config import *
from activestorage.dummy_data import make_compressed_ncdata
from activestorage.reductionist import ReductionistError as RedErr

import utils


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
    print("Test file and storage options", test_file, storage_options)
    if not utils.get_storage_type():
        storage_options = None
        active_storage_url = None
    active = Active(test_file_uri, 'tas', utils.get_storage_type(),
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)
    active._version = 2
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

    result = active[0:2,4:6,7:9]
    assert result == 239.25946044921875


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
