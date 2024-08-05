import os
import numpy as np
import pytest

from netCDF4 import Dataset
from pathlib import Path

from activestorage.active import Active, load_from_s3
from activestorage.config import *
from activestorage.dummy_data import make_compressed_ncdata

import utils


def check_dataset_filters(temp_file: str, ncvar: str, compression: str, shuffle: bool):
    # Sanity check that test data is compressed and filtered as expected.
    if USE_S3:
        with load_from_s3(temp_file) as test_data:
            print("File attrs", test_data.attrs)
            assert test_data[ncvar].compression == "gzip"
            assert test_data[ncvar].shuffle == shuffle
    else:
        with Dataset(temp_file) as test_data:
            test_data_filters = test_data.variables[ncvar].filters()
            assert test_data_filters[compression]
            assert test_data_filters['shuffle'] == shuffle


def create_compressed_dataset(tmp_path: str, compression: str, shuffle: bool):
    """
    Make a vanilla test dataset which is compressed and optionally shuffled.
    """
    temp_file = str(tmp_path / "test_compression.nc")
    test_data = make_compressed_ncdata(filename=temp_file, compression=compression, shuffle=shuffle)
    test_data = utils.write_to_storage(temp_file)
    if USE_S3:
        os.remove(temp_file)

    check_dataset_filters(test_data, "data", compression, shuffle)
    return test_data


STORAGE_OPTIONS_CLASSIC = {
    'key': S3_ACCESS_KEY,
    'secret': S3_SECRET_KEY,
    'client_kwargs': {'endpoint_url': S3_URL},
}
S3_ACTIVE_URL_MINIO = S3_ACTIVE_STORAGE_URL
S3_ACTIVE_URL_Bryan = "https://192.171.169.248:8080"

# TODO include all supported configuration types
# so far test three possible configurations for storage_options:
# - storage_options = None, active_storage_url = None (Minio and local Reductionist, preset credentials from config.py)
# - storage_options = CLASSIC, active_storage_url = CLASSIC (Minio and local Reductionist, preset credentials from config.py but folded in storage_options and active_storage_url)
# - storage_options = CLASSIC, active_storage_url = Bryan's machine (Minio BUT Reductionist moved on Bryan's machine)
#   (this invariably fails due to data URL being //localhost:9000 closed to outside Reductionist
storage_options_paramlist = [
    (None, None),
    (STORAGE_OPTIONS_CLASSIC, S3_ACTIVE_URL_MINIO),
#    (STORAGE_OPTIONS_CLASSIC, S3_ACTIVE_URL_Bryan)
]


@pytest.mark.parametrize('compression', ['zlib'])
@pytest.mark.parametrize('shuffle', [False, True])
def test_compression_and_filters(tmp_path: str, compression: str, shuffle: bool):
    """
    Test use of datasets with compression and filters applied.
    """
    test_file = create_compressed_dataset(tmp_path, compression, shuffle)

    active = Active(test_file, 'data', utils.get_storage_type())
    active._version = 1
    active._method = "min"
    result = active[0:2,4:6,7:9]
    assert result == 740.0


@pytest.mark.parametrize("storage_options, active_storage_url", storage_options_paramlist)
def test_compression_and_filters_cmip6_data(storage_options, active_storage_url):
    """
    Test use of datasets with compression and filters applied for a real
    CMIP6 dataset (CMIP6_IPSL-CM6A-LR_tas).
    """
    test_file = str(Path(__file__).resolve().parent / 'test_data' / 'CMIP6_IPSL-CM6A-LR_tas.nc')
    with Dataset(test_file) as nc_data:
        nc_min = np.min(nc_data["tas"][0:2,4:6,7:9])
    print(f"Numpy min from compressed file {nc_min}")

    test_file = utils.write_to_storage(test_file)

    check_dataset_filters(test_file, "tas", "zlib", False)

    print("Test file and storage options", test_file, storage_options)
    if not utils.get_storage_type():
        storage_options = None
        active_storage_url = None
    active = Active(test_file, 'tas', utils.get_storage_type(),
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)
    active._version = 1
    active._method = "min"
    result = active[0:2,4:6,7:9]
    assert nc_min == result
    assert result == 239.25946044921875


@pytest.mark.parametrize("storage_options, active_storage_url", storage_options_paramlist)
def test_compression_and_filters_obs4mips_data(storage_options, active_storage_url):
    """
    Test use of datasets with compression and filters applied for a real
    obs4mips dataset (obs4MIPS_CERES-EBAF_L3B_Ed2-8_rlut.nc) at CMIP5 MIP standard
    but with CMIP6-standard file packaging.
    """
    test_file = str(Path(__file__).resolve().parent / 'test_data' / 'obs4MIPS_CERES-EBAF_L3B_Ed2-8_rlut.nc')
    with Dataset(test_file) as nc_data:
        nc_min = np.min(nc_data["rlut"][0:2,4:6,7:9])
    print(f"Numpy min from compressed file {nc_min}")

    test_file = utils.write_to_storage(test_file)

    check_dataset_filters(test_file, "rlut", "zlib", False)

    print("Test file and storage options", test_file, storage_options)
    if not utils.get_storage_type():
        storage_options = None
        active_storage_url = None
    active = Active(test_file, 'rlut', utils.get_storage_type(),
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)
    active._version = 1
    active._method = "min"
    result = active[0:2,4:6,7:9]
    print(nc_min)
    assert nc_min == result
    assert result == 124.0
