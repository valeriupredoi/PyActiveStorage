import os
import numpy as np
import pytest

import activestorage.storage as st


def test_reduce_chunk():
    """Test reduce chunk entirely."""
    rfile = "tests/test_data/cesm2_native.nc"
    offset = 2
    size = 128

    # no compression
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=[None, 2050, None, None],
                         dtype="i2", shape=(8, 8),
                         order="C", chunk_selection=slice(0, 2, 1),
                         method=np.min)
    assert rc[0] == -1
    assert rc[1] == 15


def test_reduced_chunk_masked_data():
    """Test method with masked data."""
    rfile = "tests/test_data/daily_data_masked.nc"
    offset = 6911
    size = 2976

    # no compression
    ch_sel = (slice(0, 62, 1), slice(0, 2, 1),
              slice(0, 3, 1), slice(0, 2, 1))
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=(None, 999.0, None, None),
                         dtype="float32", shape=(62, 2, 3, 2),
                         order="C", chunk_selection=ch_sel,
                         method=np.mean)
    # test the output dtype
    np.testing.assert_raises(AssertionError,
                             np.testing.assert_array_equal, rc, (249.459564, 680))
    # test result with correct dtype
    np.testing.assert_array_equal(rc, (np.float32(249.459564), 680))


def test_reduced_chunk_fully_masked_data_fill():
    """Test method with masked data."""
    rfile = "tests/test_data/daily_data_fullmask.nc"
    offset = 6911
    size = 2976

    # no compression
    ch_sel = (slice(0, 62, 1), slice(0, 2, 1),
              slice(0, 3, 1), slice(0, 2, 1))
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=(None, 999.0, None, None),
                         dtype="float32", shape=(62, 2, 3, 2),
                         order="C", chunk_selection=ch_sel,
                         method=np.mean)
    assert rc[0].size == 0
    assert rc[1] == 0


def test_reduced_chunk_fully_masked_data_missing():
    """Test method with masked data."""
    rfile = "tests/test_data/daily_data_fullmask.nc"
    offset = 6911
    size = 2976

    # no compression
    ch_sel = (slice(0, 62, 1), slice(0, 2, 1),
              slice(0, 3, 1), slice(0, 2, 1))
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=(999., None, None, None),
                         dtype="float32", shape=(62, 2, 3, 2),
                         order="C", chunk_selection=ch_sel,
                         method=np.mean)
    assert rc[0].size == 0
    assert rc[1] == 0


def test_reduced_chunk_fully_masked_data_vmin():
    """Test method with masked data."""
    rfile = "tests/test_data/daily_data_fullmask.nc"
    offset = 6911
    size = 2976

    # no compression
    ch_sel = (slice(0, 62, 1), slice(0, 2, 1),
              slice(0, 3, 1), slice(0, 2, 1))
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=(None, None, 1000., None),
                         dtype="float32", shape=(62, 2, 3, 2),
                         order="C", chunk_selection=ch_sel,
                         method=np.mean)
    assert rc[0].size == 0
    assert rc[1] == 0


def test_reduced_chunk_fully_masked_data_vmax():
    """Test method with masked data."""
    rfile = "tests/test_data/daily_data_fullmask.nc"
    offset = 6911
    size = 2976

    # no compression
    ch_sel = (slice(0, 62, 1), slice(0, 2, 1),
              slice(0, 3, 1), slice(0, 2, 1))
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=(None, None, None, 1.),
                         dtype="float32", shape=(62, 2, 3, 2),
                         order="C", chunk_selection=ch_sel,
                         method=np.mean)
    assert rc[0].size == 0
    assert rc[1] == 0


def test_zero_data():
    """Test method with zero data."""
    rfile = "tests/test_data/zero_chunked.nc"
    offset = 8760
    size = 48

    # no compression
    ch_sel = (slice(0, 3, 1), slice(0, 4, 1))
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=(None, None, None, None),
                         dtype="float32", shape=(3, 4),
                         order="C", chunk_selection=ch_sel,
                         method=np.mean)
    assert rc[0].size == 1
    assert rc[0] == 0
    assert rc[1] is 12
