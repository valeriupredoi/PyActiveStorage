
from activestorage import dummy_data as dd
from activestorage import Active
import pyfive
import numpy.ma as ma
import numpy as np
import os
from activestorage.config import *
from pathlib import Path
import s3fs


def get_masked_data(ds):
    """ This is buried in active itself we should pull it out for wider usage in our testing etc"""
    attrs = ds.attrs
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

    data = ds[:]
    
    if _FillValue is not None:
        data = np.ma.masked_equal(data, _FillValue)

    if missing_value is not None:
        data = np.ma.masked_equal(data, missing_value)

    if valid_max is not None:
        data = np.ma.masked_greater(data, valid_max)

    if valid_min is not None:
        data = np.ma.masked_less(data, valid_min)

    return data




def upload_to_s3(server, username, password, bucket, object, rfile):
    """Upload a file to an S3 object store."""
    s3_fs = s3fs.S3FileSystem(key=username, secret=password, client_kwargs={'endpoint_url': server})
    # Make sure s3 bucket exists
    try:
        s3_fs.mkdir(bucket)
    except FileExistsError:
        pass

    s3_fs.put_file(rfile, os.path.join(bucket, object))


def get_storage_type():
    if USE_S3:
        return "s3"
    else:
        return None
    
def write_to_storage(ncfile):
    """Write a file to storage and return an appropriate URI or path to access it."""
    if USE_S3:
        object = os.path.basename(ncfile)
        upload_to_s3(S3_URL, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET, object, ncfile)
        return os.path.join("s3://", S3_BUCKET, object)
    else:
        return ncfile


def active_zero(testfile):
    """Run Active with no active storage (version=0)."""
    active = Active(testfile, "data", get_storage_type())
    active._version = 0
    d = active[0:2, 4:6, 7:9]
    assert ma.is_masked(d)

    # FIXME: For the S3 backend, h5netcdf is used to read the metadata. It does
    # not seem to load the missing data attributes (missing_value, _FillValue,
    # valid_min, valid_max, valid_range, etc).
    assert ma.is_masked(d)

    return np.mean(d)


def active_two(testfile):
    """Run Active with active storage (version=2)."""
    active = Active(testfile, "data", get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]

    active_mean = result2["sum"] / result2["n"]

    return active_mean


def load_dataset(testfile):
    """Load data as netCDF4.Dataset."""
    ds = pyfive.File(testfile)
    actual_data = get_masked_data(ds["data"])
    
    ds.close()

    assert ma.is_masked(actual_data)

    return actual_data


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
    testfile = write_to_storage(testfile)

    # numpy masked to check for correct Active behaviour
    no_active_mean = active_zero(testfile)
    print("No active storage result (mean)", no_active_mean)

    active_mean = active_two(testfile)
    print("Active storage result (mean)", active_mean)

    np.testing.assert_array_equal(masked_numpy_mean, active_mean)
    np.testing.assert_array_equal(no_active_mean, active_mean)
