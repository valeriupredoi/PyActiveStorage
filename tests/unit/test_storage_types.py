# NOTE: Since these are unit tests, we can't assume that an S3 object store or
# active storage server is available. Therefore, we mock out the remote service
# interaction and replace with local file operations.

import botocore
import contextlib
import os
import h5netcdf
import numpy as np
import pytest
import requests.exceptions
from unittest import mock


import activestorage.active
from activestorage.active import Active
from activestorage.config import *
from activestorage.dummy_data import make_vanilla_ncdata
from activestorage import netcdf_to_zarr
import activestorage.reductionist
import activestorage.storage


# Capture the real function before it is mocked.
old_netcdf_to_zarr = netcdf_to_zarr.load_netcdf_zarr_generic

storage_options = {
    'key': S3_ACCESS_KEY,
    'secret': S3_SECRET_KEY,
    'client_kwargs': {'endpoint_url': S3_URL},
}
     
@mock.patch.object(activestorage.active, "load_from_s3")
@mock.patch.object(activestorage.netcdf_to_zarr, "load_netcdf_zarr_generic")
@mock.patch.object(activestorage.active.reductionist, "reduce_chunk")
def test_s3(mock_reduce, mock_nz, mock_load, tmp_path):
    """Test stack when call to Active contains storage_type == s3."""

    # Since this is a unit test, we can't assume that an S3 object store or
    # active storage server is available. Therefore, we mock out the remote
    # service interaction and replace with local file operations.

    @contextlib.contextmanager
    def load_from_s3(uri, storage_options):
        yield h5netcdf.File(test_file, 'r', invalid_netcdf=True)

    def load_netcdf_zarr_generic(uri, ncvar, storage_type, storage_options):
        return old_netcdf_to_zarr(test_file, ncvar, storage_options, None)

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
    mock_nz.side_effect = load_netcdf_zarr_generic
    mock_reduce.side_effect = reduce_chunk

    uri = "s3://fake-bucket/fake-object"
    test_file = str(tmp_path / "test.nc")
    make_vanilla_ncdata(test_file)

    active = Active(
        uri, "data", storage_options=storage_options,
        active_storage_url=S3_ACTIVE_STORAGE_URL)
    active._version = 1
    active._method = "max"

    result = active[::]

    assert result == 999.0

    mock_load.assert_called_once_with(uri)
    mock_nz.assert_called_once_with(
        uri, "data",
        storage_options=storage_options,
        active_storage_url=S3_ACTIVE_STORAGE_URL
    )
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

    @contextlib.contextmanager
    def load_from_s3(uri, storage_options):
        yield h5netcdf.File(test_file, 'r', invalid_netcdf=True)

    mock_load.side_effect = load_from_s3

    uri = "s3://fake-bucket/fake-object"
    test_file = str(tmp_path / "test.nc")
    make_vanilla_ncdata(test_file)

    active = Active(uri, "data", storage_options=storage_options)
    active._version = 0

    result = active[::]

    assert np.max(result) == 999.0


@mock.patch.object(activestorage.active, "load_from_s3")
def test_s3_load_failure(mock_load):
    """Test when an S3 object doesn't exist."""
    uri = "s3://fake-bucket/fake-object"

    mock_load.side_effect = FileNotFoundError

    with pytest.raises(FileNotFoundError):
        Active(uri, "data", storage_options=storage_options)


@mock.patch.object(activestorage.active, "load_from_s3")
@mock.patch.object(activestorage.netcdf_to_zarr, "load_netcdf_zarr_generic")
@mock.patch.object(activestorage.active.reductionist, "reduce_chunk")
def test_reductionist_connection(mock_reduce, mock_nz, mock_load, tmp_path):
    """Test stack when call to Active contains storage_type == s3."""

    @contextlib.contextmanager
    def load_from_s3(uri, storage_options):
        yield h5netcdf.File(test_file, 'r', invalid_netcdf=True)

    def load_netcdf_zarr_generic(uri, ncvar, storage_type, storage_options):
        return old_netcdf_to_zarr(test_file, ncvar, storage_options, None)

    mock_load.side_effect = load_from_s3
    mock_nz.side_effect = load_netcdf_zarr_generic
    mock_reduce.side_effect = requests.exceptions.ConnectTimeout()

    uri = "s3://fake-bucket/fake-object"
    test_file = str(tmp_path / "test.nc")
    make_vanilla_ncdata(test_file)

    active = Active(uri, "data", storage_options=storage_options,
                    active_storage_url=S3_ACTIVE_STORAGE_URL)
    active._version = 1
    active._method = "max"

    with pytest.raises(requests.exceptions.ConnectTimeout):
        assert active[::]


@mock.patch.object(activestorage.active, "load_from_s3")
@mock.patch.object(activestorage.netcdf_to_zarr, "load_netcdf_zarr_generic")
@mock.patch.object(activestorage.active.reductionist, "reduce_chunk")
def test_reductionist_bad_request(mock_reduce, mock_nz, mock_load, tmp_path):
    """Test stack when call to Active contains storage_type == s3."""

    @contextlib.contextmanager
    def load_from_s3(uri, storage_options):
        yield h5netcdf.File(test_file, 'r', invalid_netcdf=True)

    def load_netcdf_zarr_generic(uri, ncvar, storage_type, storage_options):
        return old_netcdf_to_zarr(test_file, ncvar, storage_options, None)

    mock_load.side_effect = load_from_s3
    mock_nz.side_effect = load_netcdf_zarr_generic
    mock_reduce.side_effect = activestorage.reductionist.ReductionistError(400, "Bad request")

    uri = "s3://fake-bucket/fake-object"
    test_file = str(tmp_path / "test.nc")
    make_vanilla_ncdata(test_file)

    active = Active(uri, "data", storage_options=storage_options)
    active._version = 1
    active._method = "max"

    with pytest.raises(activestorage.reductionist.ReductionistError):
        assert active[::]
