import os
import numpy as np
import pytest
import shutil
import tempfile
import unittest

from activestorage.active import Active
from activestorage.config import *
from activestorage.dummy_data import make_vanilla_ncdata

import utils


def create_test_dataset(tmp_path):
    """
    Ensure there is test data
    """
    temp_file = str(tmp_path / 'test_bizarre.nc')
    make_vanilla_ncdata(filename=temp_file)
    test_file = utils.write_to_storage(temp_file)
    if USE_S3:
        os.remove(temp_file)
    return test_file


# @pytest.mark.xfail(USE_S3, reason="descriptor 'flatten' for 'numpy.ndarray' objects doesn't apply to a 'memoryview' object")
def test_read0(tmp_path):
    """
    Test a normal read slicing the data an interesting way, using version 0 (native interface)
    """
    test_file = create_test_dataset(tmp_path)
    active = Active(test_file, 'data', utils.get_storage_type())
    active._version = 0
    d = active[0:2, 4:6, 7:9]
    # d.data is a memoryview object in both local POSIX and remote S3 storages
    # keep the current behaviour of the test to catch possible type changes
    nda = np.ndarray.flatten(np.asarray(d.data))
    assert np.array_equal(nda,np.array([740.,840.,750.,850.,741.,841.,751.,851.]))

def test_read1(tmp_path):
    """
    Test a normal read slicing the data an interesting way, using version 1 (replicating native interface in our code)
    """
    test_file = create_test_dataset(tmp_path)
    active = Active(test_file, 'data', utils.get_storage_type())
    active._version = 0
    d0 = active[0:2,4:6,7:9]

    active = Active(test_file, 'data', utils.get_storage_type())
    active._version = 1
    d1 = active[0:2,4:6,7:9]
    assert np.array_equal(d0,d1)

def test_active(tmp_path):
    """
    Shows what we expect an active example test to achieve and provides "the right answer"
    """
    test_file = create_test_dataset(tmp_path)
    active = Active(test_file, 'data', utils.get_storage_type())
    active._version = 0
    d = active[0:2,4:6,7:9]
    mean_result = np.mean(d)

    active = Active(test_file, 'data', utils.get_storage_type())
    active.method = "mean"
    result2 = active[0:2,4:6,7:9]
    assert mean_result == result2

def testActiveComponents(tmp_path):
    """
    Shows what we expect an active example test to achieve and provides "the right answer"
    """
    test_file = create_test_dataset(tmp_path)
    active = Active(test_file, "data", utils.get_storage_type())
    active._version = 0
    d = active[0:2, 4:6, 7:9]
    mean_result = np.mean(d)

    active = Active(test_file, "data", utils.get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]
    print(result2)
    assert mean_result == result2["sum"]/result2["n"]
