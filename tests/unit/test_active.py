import os
import numpy as np
import pytest
import threading

from activestorage.active import Active
from activestorage.active import load_from_s3
from activestorage.config import *
from botocore.exceptions import EndpointConnectionError as botoExc
from botocore.exceptions import NoCredentialsError as NoCredsExc


def test_uri_none():
    """Unit test for class:Active."""
    # test invalid uri
    some_file = None
    expected = "Must use a valid file for uri. Got None"
    with pytest.raises(ValueError) as exc:
        active = Active(some_file, ncvar="")
    assert str(exc.value) == expected


def test_uri_nonexistent():
    """Unit test for class:Active."""
    # test invalid uri
    some_file = 'cow.nc'
    expected = "Must use existing file for uri. cow.nc not found"
    with pytest.raises(ValueError) as exc:
        active = Active(some_file, ncvar="")
    assert str(exc.value) == expected


def test_getitem():
    """Unit test for class:Active."""
    # no variable passed
    uri = "tests/test_data/emac.nc"
    index = 3
    with pytest.raises(ValueError) as exc:
        active = Active(uri, ncvar=None)
    assert str(exc.value) == "Must set a netCDF variable name to slice"

    # openable file and correct variable
    uri = "tests/test_data/cesm2_native.nc"
    ncvar = "TREFHT"
    active = Active(uri, ncvar=ncvar)
    assert active._version == 1
    item = active.__getitem__(index)[0, 0]
    expected = np.array(277.11945, dtype="float32")
    np.testing.assert_array_equal(item, expected)

    # test version iterations
    active._version = 0
    assert active._version == 0
    assert active.method is None
    item = active.__getitem__(index)[0, 0]
    expected = np.array(277.11945, dtype="float32")
    np.testing.assert_array_equal(item, expected)
    active._version = 3
    with pytest.raises(ValueError) as exc:
        item = active.__getitem__(index)[0, 0]
    assert str(exc.value) == "Version 3 not supported"


def test_method():
    """Unit test for class:Active."""
    uri = "tests/test_data/cesm2_native.nc"
    ncvar = "TREFHT"
    active = Active(uri, ncvar=ncvar)
    active._method = "min"
    assert active.method([3,444]) == 3

    active._method = "cow"
    assert active.method is None


def test_active():
    """Test with full complement of args."""
    uri = "tests/test_data/cesm2_native.nc"
    ncvar = "TREFHT"
    active = Active(uri, ncvar=ncvar)
    init = active.__init__(uri=uri, ncvar=ncvar)


@pytest.mark.xfail(reason="We don't employ locks with Pyfive anymore, yet.")
def test_lock():
    """Unit test for class:Active."""
    uri = "tests/test_data/cesm2_native.nc"
    ncvar = "TREFHT"
    active = Active(uri, ncvar=ncvar)

    lock = threading.Lock()
    active.lock = lock
    assert active.lock is lock

    # Pass through code that uses the lock
    active.method = None
    index = 3
    for version in (0, 2):
        active._version = version
        active[index]

    active.lock = None
    assert active.lock is False


@pytest.mark.skipif(USE_S3 == True, reason="it will look for silly bucket")
def test_load_from_s3():
    """Test basic load from S3 without loading from S3."""
    uri = "s3://bucket/file.nc"
    expected_exc = "Could not connect to the endpoint URL"
    with pytest.raises(botoExc) as exc:
        with load_from_s3(uri) as nc:
            data = nc["cow"][0]
    assert expected_exc in str(exc.value)


@pytest.mark.skipif(USE_S3 == True, reason="it will look for silly bucket")
def test_load_from_s3_so_None():
    """Test basic load from S3 without loading from S3."""
    uri = "s3://bucket/file.nc"
    expected_exc = "Unable to locate credentials"
    with pytest.raises(NoCredsExc) as exc:
        with load_from_s3(uri, storage_options={}) as nc:
            data = nc["cow"][0]
    assert expected_exc in str(exc.value)


@pytest.mark.skipif(USE_S3 == True, reason="it will look for silly URIs")
def test_get_endpoint_url():
    """Test _get_endpoint_url(self) from Active class."""
    storage_options = {
        'key': "cow",
        'secret': "secretcow",
        'client_kwargs': {'endpoint_url': "https://cow.moo"},
    }
    uri = "tests/test_data/cesm2_native.nc"
    ncvar = "TREFHT"
    active = Active(uri, ncvar=ncvar, storage_options=storage_options)
    ep_url = Active._get_endpoint_url(active)
    assert ep_url == "https://cow.moo"
