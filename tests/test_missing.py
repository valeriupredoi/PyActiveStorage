import os
from this import d
import numpy as np
import numpy.ma as ma
import pytest
import shutil
import tempfile
import unittest

# TODO remove in stable
import h5py
import h5netcdf

import pyfive

from netCDF4 import Dataset

from activestorage.active import Active, load_from_s3
from activestorage.config import *
from activestorage import dummy_data as dd

import utils


def load_dataset(testfile):
    """Load data as netCDF4.Dataset."""
    ds = Dataset(testfile)
    actual_data = ds["data"][:]
    ds.close()

    assert ma.is_masked(actual_data)

    return actual_data


def active_zero(testfile):
    """Run Active with no active storage (version=0)."""
    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 0
    d = active[0:2, 4:6, 7:9]

    # FIXME: For the S3 backend, h5netcdf is used to read the metadata. It does
    # not seem to load the missing data attributes (missing_value, _FillValue,
    # valid_min, valid_max, valid_range, etc).
    assert ma.is_masked(d)

    return np.mean(d)


def active_two(testfile):
    """Run Active with active storage (version=2)."""
    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]

    active_mean = result2["sum"] / result2["n"]

    return active_mean


def test_partially_missing_data(tmp_path):
    testfile = str(tmp_path / 'test_partially_missing_data.nc')
    r = dd.make_partially_missing_ncdata(testfile)

    # retrieve the actual numpy-ed result
    actual_data = load_dataset(testfile)
    unmasked_numpy_mean = actual_data[0:2, 4:6, 7:9].data.mean()
    masked_numpy_mean = actual_data[0:2, 4:6, 7:9].mean()
    assert unmasked_numpy_mean != masked_numpy_mean
    print("Numpy masked result (mean)", masked_numpy_mean)

    # write file to storage
    testfile = utils.write_to_storage(testfile)

    # numpy masked to check for correct Active behaviour
    no_active_mean = active_zero(testfile)
    print("No active storage result (mean)", no_active_mean)

    active_mean = active_two(testfile)
    print("Active storage result (mean)", active_mean)

    np.testing.assert_array_equal(masked_numpy_mean, active_mean)
    np.testing.assert_array_equal(no_active_mean, active_mean)


def test_missing(tmp_path):
    testfile = str(tmp_path / 'test_missing.nc')
    r = dd.make_missing_ncdata(testfile)

    # retrieve the actual numpy-ed result
    actual_data = load_dataset(testfile)
    unmasked_numpy_mean = actual_data[0:2, 4:6, 7:9].data.mean()
    masked_numpy_mean = actual_data[0:2, 4:6, 7:9].mean()
    assert unmasked_numpy_mean != masked_numpy_mean
    print("Numpy masked result (mean)", masked_numpy_mean)

    # write file to storage
    testfile = utils.write_to_storage(testfile)

    # numpy masked to check for correct Active behaviour
    no_active_mean = active_zero(testfile)
    print("No active storage result (mean)", no_active_mean)

    active_mean = active_two(testfile)
    print("Active storage result (mean)", active_mean)

    np.testing.assert_array_equal(masked_numpy_mean, active_mean)
    np.testing.assert_array_equal(no_active_mean, active_mean)



def test_fillvalue(tmp_path):
    """
    fill_value set to -999 from dummy_data.py
    note: no _FillValue attr set here, just fill_value!
    """
    testfile = str(tmp_path / 'test_fillvalue.nc')
    r = dd.make_fillvalue_ncdata(testfile)

    # retrieve the actual numpy-ed result
    actual_data = load_dataset(testfile)
    unmasked_numpy_mean = actual_data[0:2, 4:6, 7:9].data.mean()
    masked_numpy_mean = actual_data[0:2, 4:6, 7:9].mean()
    assert unmasked_numpy_mean != masked_numpy_mean
    print("Numpy masked result (mean)", masked_numpy_mean)

    # write file to storage
    testfile = utils.write_to_storage(testfile)

    # numpy masked to check for correct Active behaviour
    no_active_mean = active_zero(testfile)
    print("No active storage result (mean)", no_active_mean)

    active_mean = active_two(testfile)
    print("Active storage result (mean)", active_mean)

    np.testing.assert_array_equal(masked_numpy_mean, active_mean)
    np.testing.assert_array_equal(no_active_mean, active_mean)


def test_validmin(tmp_path):
    """
    Validmin, validmax, and validrange cases apply the upper, lower,
    or interval limits after chunk selection, and the application is
    per chunk data, so it is important to have a test
    that knows what is the min, max, and interval for the selected data,
    otherwise the test is futile!

    In this test data is constructed with a validmin of 200., but the selected
    chunks all have data >=750., so we apply a validmin == 751.
    """
    testfile = str(tmp_path / 'test_validmin.nc')
    r = dd.make_validmin_ncdata(testfile, valid_min=751.)

    # retrieve the actual numpy-ed result
    actual_data = load_dataset(testfile)
    unmasked_numpy_mean = actual_data[0:2, 4:6, 7:9].data.mean()
    masked_numpy_mean = actual_data[0:2, 4:6, 7:9].mean()
    assert unmasked_numpy_mean != masked_numpy_mean
    print("Numpy masked result (mean)", masked_numpy_mean)

    # write file to storage
    testfile = utils.write_to_storage(testfile)

    # numpy masked to check for correct Active behaviour
    no_active_mean = active_zero(testfile)
    print("No active storage result (mean)", no_active_mean)

    active_mean = active_two(testfile)
    print("Active storage result (mean)", active_mean)

    np.testing.assert_array_equal(masked_numpy_mean, active_mean)
    np.testing.assert_array_equal(no_active_mean, active_mean)


def test_validmax(tmp_path):
    """
    Validmin, validmax, and validrange cases apply the upper, lower,
    or interval limits after chunk selection, and the application is
    per chunk data, so it is important to have a test
    that knows what is the min, max, and interval for the selected data,
    otherwise the test is futile!

    In this test data is constructed with a validmin of 200., but the selected
    chunks all have data >=750., so we apply a validmax == 850.
    """
    testfile = str(tmp_path / 'test_validmax.nc')
    r = dd.make_validmax_ncdata(testfile, valid_max=850.)

    # retrieve the actual numpy-ed result
    actual_data = load_dataset(testfile)
    unmasked_numpy_mean = np.mean(actual_data[0:2, 4:6, 7:9])
    print("Numpy masked result (mean)", unmasked_numpy_mean)
    unmasked_numpy_mean = actual_data[0:2, 4:6, 7:9].data.mean()
    masked_numpy_mean = actual_data[0:2, 4:6, 7:9].mean()
    assert unmasked_numpy_mean != masked_numpy_mean
    print("Numpy masked result (mean)", masked_numpy_mean)

    # load files via external protocols
    y = Dataset(testfile)
    z = h5py.File(testfile)
    a = h5netcdf.File(testfile)

    # write file to storage
    testfile = utils.write_to_storage(testfile)

    # load file via our protocols
    if USE_S3:
        x = load_from_s3(testfile)
    else:
        x = pyfive.File(testfile)

    # print for Bryan
    print('bnl',y['data'].getncattr('valid_max'))
    print('bnl',x['data'].attrs.get('valid_max'))
    print('bnl',z['data'].attrs.get('valid_max'))
    print('bnl',a['data'].attrs.get('valid_max'))



    # numpy masked to check for correct Active behaviour
    no_active_mean = active_zero(testfile)

    print("No active storage result (mean)", no_active_mean)

    active_mean = active_two(testfile)
    print("Active storage result (mean)", active_mean)

    np.testing.assert_array_equal(masked_numpy_mean, active_mean)
    np.testing.assert_array_equal(no_active_mean, active_mean)


def test_validrange(tmp_path):
    """
    Validmin, validmax, and validrange cases apply the upper, lower,
    or interval limits after chunk selection, and the application is
    per chunk data, so it is important to have a test
    that knows what is the min, max, and interval for the selected data,
    otherwise the test is futile!

    In this test data is constructed with a validmin of 200., but the selected
    chunks all have data >=750. and <=851., so we apply a validrange == [750, 850.]
    """
    testfile = str(tmp_path / 'test_validrange.nc')
    r = dd.make_validrange_ncdata(testfile, valid_range=[750., 850.])

    # retrieve the actual numpy-ed result
    actual_data = load_dataset(testfile)
    unmasked_numpy_mean = actual_data[0:2, 4:6, 7:9].data.mean()
    masked_numpy_mean = actual_data[0:2, 4:6, 7:9].mean()
    assert unmasked_numpy_mean != masked_numpy_mean
    print("Numpy masked result (mean)", masked_numpy_mean)

    # write file to storage
    testfile = utils.write_to_storage(testfile)

    # numpy masked to check for correct Active behaviour
    no_active_mean = active_zero(testfile)
    print("No active storage result (mean)", no_active_mean)

    active_mean = active_two(testfile)
    print("Active storage result (mean)", active_mean)

    np.testing.assert_array_equal(masked_numpy_mean, active_mean)
    np.testing.assert_array_equal(no_active_mean, active_mean)


def test_active_mask_data(tmp_path):
    testfile = str(tmp_path / 'test_partially_missing_data.nc')

    def check_masking(testfile, testname):

        valid_masked_data = load_dataset(testfile)
        ds = pyfive.File(testfile)
        dsvar = ds["data"]
        dsdata = dsvar[:]
        ds.close()
        a = Active(testfile, "data")
        data = a._mask_data(dsdata)
        np.testing.assert_array_equal(data, valid_masked_data,f'Failed masking for {testname}')

    # with valid min
    r = dd.make_validmin_ncdata(testfile, valid_min=500)
    check_masking(testfile, "valid min")

    # with valid range
    r = dd.make_validrange_ncdata(testfile, valid_range=[750., 850.])
    check_masking(testfile, "valid range")

    # retrieve the actual numpy-ed result
    actual_data = load_dataset(testfile)

    # with missing
    r = dd.make_missing_ncdata(testfile)
    check_masking(testfile,'missing')

    # with _FillValue
    r = dd.make_fillvalue_ncdata(testfile)
    check_masking(testfile,"_FillValue")

