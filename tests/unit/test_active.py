import os
import numpy as np
import pytest
import tempfile

from activestorage.active import Active


def test_no_attrs():
    """Unit test for class:Active."""
    # test lack of init
    expected = "Active.__init__() missing 1 required positional argument: 'uri'"
    with pytest.raises(TypeError) as exc:
        active = Active()
    assert str(exc.value) == expected


def test_uri_none():
    """Unit test for class:Active."""
    # test invalid uri
    some_file = None
    expected = "Must use a valid file for uri. Got None"
    with pytest.raises(ValueError) as exc:
        active = Active(some_file)
    assert str(exc.value) == expected


def test_uri_nonexistent():
    """Unit test for class:Active."""
    # test invalid uri
    some_file = 'cow.nc'
    expected = "Must use existing file for uri. cow.nc not found"
    with pytest.raises(ValueError) as exc:
        active = Active(some_file)
    assert str(exc.value) == expected
