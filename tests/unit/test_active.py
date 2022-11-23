import os
import numpy as np
import pytest

from activestorage.active import Active


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

    # unopenable file
    ncvar = "tas"
    baseexc = "tas not found in /"
    with pytest.raises(IndexError) as exc:
        active = Active(uri, ncvar=ncvar)
    assert baseexc in str(exc.value)

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
    init = active.__init__(uri=uri, ncvar=ncvar, missing_value=True,
                           fill_value=1e20, valid_min=-1,
                           valid_max=1200)


def test_config_s3():
    uri = "tests/test_data/cesm2_native.nc"
    ncvar = "TREFHT"
    active = Active(uri, ncvar=ncvar, storage_type="S3")
    assert active._methods == {'max': 'max', 'mean': 'mean',
                               'min': 'min', 'sum': 'dimsum'}
    assert active.method is None
    assert active._version == 1

    active._version = 2

    # statistical method can not be executed
    active.method = "mean"
    with pytest.raises(ValueError) as exc:
        active[:]
    assert str(exc.value) == "Could not recognize method mean as permitted."

    # bad name for statistical method
    with pytest.raises(ValueError) as exc:
        active.method = "meany"
    assert str(exc.value) == "Bad 'method': meany. Choose from min/max/mean/sum."


def test_config_Posix():
    uri = "tests/test_data/cesm2_native.nc"
    ncvar = "TREFHT"
    active = Active(uri, ncvar=ncvar, storage_type="Posix")
    assert active._methods == {'max': 'np.max', 'mean': 'np.sum',
                               'min': 'np.min', 'sum': 'np.sum'}
    assert active.method is None
    assert active._version == 1

    active._version = 2

    # usual run
    active.method = "mean"  # will exec np.mean from config
    assert active[:] == 284.22694905598956

    # passing wrong numpy method
    active._methods["mean"] = "np.meany"
    with pytest.raises(AttributeError) as exc:
        active[:]
    assert str(exc.value) == "Method np.meany is not a valid Numpy method."


def test_config_invalid_storage_type():
    uri = "tests/test_data/cesm2_native.nc"
    ncvar = "TREFHT"
    with pytest.raises(ValueError) as exc:
        Active(uri, ncvar=ncvar, storage_type="cowabunga")
    assert str(exc.value) == "Storage type cowabunga not known."
