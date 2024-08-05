# NOTE: Since these are unit tests, we can't assume that an S3 object store or
# active storage server is available. Therefore, we mock out the remote service
# interaction and replace with local file operations.

import botocore
import os
import numpy as np
import pyfive
import pytest
import requests.exceptions
from unittest import mock


import activestorage.active
from activestorage.active import Active
from activestorage.config import *
from activestorage.dummy_data import make_vanilla_ncdata
import activestorage.reductionist
import activestorage.storage




@mock.patch.object(activestorage.active, "load_from_s3")
@mock.patch.object(activestorage.active.reductionist, "reduce_chunk")
def test_s3(mock_reduce, mock_load, tmp_path):
    """Test stack when call to Active contains storage_type == s3."""

    # Since this is a unit test, we can't assume that an S3 object store or
    # active storage server is available. Therefore, we mock out the remote
    # service interaction and replace with local file operations.

    def load_from_s3(uri, storage_options=None):
        return pyfive.File(test_file)

    def reduce_chunk(
        session,
        server,
        source,
        bucket,
        object,
        offset,
        size,
        compressor,
        filters,
        missing,
        dtype,
        shape,
        order,
        chunk_selection,
        operation,
    ):
        return activestorage.storage.reduce_chunk(
            test_file,
            offset,
            size,
            compressor,
            filters,
            missing,
            dtype,
            shape,
            order,
            chunk_selection,
            np.max,
        )

    mock_load.side_effect = load_from_s3
    mock_reduce.side_effect = reduce_chunk

    uri = "s3://fake-bucket/fake-object"
    test_file = str(tmp_path / "test.nc")
    make_vanilla_ncdata(test_file)

    active = Active(uri, "data", "s3")
    active._version = 2
    active._method = "max"

    print("This test has severe flakiness:")
    print("Either fails with AssestionError - bTREE stuff,")
    print("or it fails with a multitude of KeyErrors.")
    print(active)
    result = active[::]

    assert result == 999.0

    # S3 loading is done from Active
    # but we should delegate that outside at some point
    # mock_load.assert_not_called()

    # NOTE: This gets called multiple times with various arguments. Match on
    # the common ones.
    mock_reduce.assert_called_with(
        mock.ANY,
        S3_ACTIVE_STORAGE_URL,
        S3_URL,
        mock.ANY,
        mock.ANY,
        mock.ANY,
        mock.ANY,
        None,
        None,
        (None, None, None, None),
        np.dtype("float64"),
        mock.ANY,
        "C",
        mock.ANY,
        operation="max",
    )


@mock.patch.object(activestorage.active, "load_from_s3")
def test_reductionist_version_0(mock_load, tmp_path):
    """Test stack when call to Active contains storage_type == s3 using version 0."""

    def load_from_s3(uri, storage_options=None):
        return pyfive.File(test_file)

    mock_load.side_effect = load_from_s3

    uri = "s3://fake-bucket/fake-object"
    test_file = str(tmp_path / "test.nc")
    make_vanilla_ncdata(test_file)

    active = Active(uri, "data", "s3")
    active._version = 0

    result = active[::]

    assert np.max(result) == 999.0


@pytest.mark.skip(reason="No more valid file load in Active")
@mock.patch.object(activestorage.active, "load_from_s3")
def test_s3_load_failure(mock_load):
    """Test when an S3 object doesn't exist."""
    uri = "s3://fake-bucket/fake-object"

    mock_load.side_effect = FileNotFoundError

    with pytest.raises(FileNotFoundError):
        Active(uri, "data", "s3")


@mock.patch.object(activestorage.active, "load_from_s3")
@mock.patch.object(activestorage.active.reductionist, "reduce_chunk")
def test_reductionist_connection(mock_reduce, mock_load, tmp_path):
    """Test stack when call to Active contains storage_type == s3."""

    def load_from_s3(uri, storage_options=None):
        return pyfive.File(test_file)

    mock_load.side_effect = load_from_s3
    mock_reduce.side_effect = requests.exceptions.ConnectTimeout()

    uri = "s3://fake-bucket/fake-object"
    test_file = str(tmp_path / "test.nc")
    make_vanilla_ncdata(test_file)

    active = Active(uri, "data", "s3")
    active._version = 2
    active._method = "max"

    with pytest.raises(requests.exceptions.ConnectTimeout):
        assert active[::]


@mock.patch.object(activestorage.active, "load_from_s3")
@mock.patch.object(activestorage.active.reductionist, "reduce_chunk")
def test_reductionist_bad_request(mock_reduce, mock_load, tmp_path):
    """Test stack when call to Active contains storage_type == s3."""

    def load_from_s3(uri, storage_options=None):
        return pyfive.File(test_file)

    mock_load.side_effect = load_from_s3
    mock_reduce.side_effect = activestorage.reductionist.ReductionistError(400, "Bad request")

    uri = "s3://fake-bucket/fake-object"
    test_file = str(tmp_path / "test.nc")
    make_vanilla_ncdata(test_file)

    active = Active(uri, "data", "s3")
    active._version = 2
    active._method = "max"

    with pytest.raises(activestorage.reductionist.ReductionistError):
        assert active[::]
