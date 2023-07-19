import os
from this import d
import numpy as np
import pytest
import shutil
import tempfile
import unittest

from netCDF4 import Dataset

from activestorage.active import Active
from activestorage.config import *
from activestorage import dummy_data as dd

import utils


def load_dataset(testfile):
    """Load data as netCDF4.Dataset."""
    ds = Dataset(testfile)
    actual_data = ds["data"][:]
    ds.close()

    return actual_data


def active_zero(testfile):
    """Run Active with no active storage (version=0)."""
    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 0
    d = active[0:2, 4:6, 7:9]

    return d


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

    # NOT numpy masked to check for correct Active behaviour
    no_active_mean = np.mean(active_zero(testfile))
    print("No active storage result (mean)", no_active_mean)

    active_mean = active_two(testfile)
    print("Active storage result (mean)", active_mean)

    if not USE_S3:
        np.testing.assert_array_equal(masked_numpy_mean, active_mean)
    else:
        np.testing.assert_array_equal(unmasked_numpy_mean, active_mean)
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

    # NOT numpy masked to check for correct Active behaviour
    no_active_mean = np.mean(active_zero(testfile))
    print("No active storage result (mean)", no_active_mean)

    active_mean = active_two(testfile)
    print("Active storage result (mean)", active_mean)

    if not USE_S3:
        np.testing.assert_array_equal(masked_numpy_mean, active_mean)
    else:
        np.testing.assert_array_equal(unmasked_numpy_mean, active_mean)
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

    d = active_zero(testfile)

    # NOT masked
    no_active_mean = np.mean(d)
    print("No active storage result (mean)", no_active_mean)

    active_mean = active_two(testfile)
    print("Active storage result (mean)", active_mean)

    if not USE_S3:
        np.testing.assert_array_equal(masked_numpy_mean, active_mean)
    else:
        np.testing.assert_array_equal(unmasked_numpy_mean, active_mean)
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

    d = active_zero(testfile)

    # NOT numpy masked to check for correct Active behaviour
    no_active_mean = np.mean(d)
    print("No active storage result (mean)", no_active_mean)

    active_mean = active_two(testfile)
    print("Active storage result (mean)", active_mean)

    if not USE_S3:
        np.testing.assert_array_equal(masked_numpy_mean, active_mean)
    else:
        np.testing.assert_array_equal(unmasked_numpy_mean, active_mean)
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

    # write file to storage
    testfile = utils.write_to_storage(testfile)

    d = active_zero(testfile)

    # NOT numpy masked to check for correct Active behaviour
    no_active_mean = np.mean(d)
    print("No active storage result (mean)", no_active_mean)

    active_mean = active_two(testfile)
    print("Active storage result (mean)", active_mean)

    if not USE_S3:
        np.testing.assert_array_equal(masked_numpy_mean, active_mean)
    else:
        np.testing.assert_array_equal(unmasked_numpy_mean, active_mean)


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

    d = active_zero(testfile)

    # NOT numpy masked to check for correct Active behaviour
    no_active_mean = np.mean(d)
    print("No active storage result (mean)", no_active_mean)

    active_mean = active_two(testfile)
    print("Active storage result (mean)", active_mean)

    if not USE_S3:
        np.testing.assert_array_equal(masked_numpy_mean, active_mean)
    else:
        np.testing.assert_array_equal(unmasked_numpy_mean, active_mean)
