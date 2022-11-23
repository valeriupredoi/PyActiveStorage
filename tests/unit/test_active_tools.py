import os
import numpy as np
import pytest
import zarr

from activestorage import active_tools as at
from numcodecs import Blosc
from zarr.indexing import (
    OrthogonalIndexer,
    is_contiguous_selection,
)
from zarr.util import is_total_slice


def assemble_zarr():
    """Create a test zarr object."""
    compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)
    z = zarr.create((10000, 10000), chunks=(1000, 1000), dtype='i1', order='C',
                    compressor=compressor)

    return z


def assemble_zarr_uncompressed():
    """Create a test zarr object."""
    z = zarr.create((1000, 1000), chunks=(2, 8), dtype='i1', order='C',
                    compressor=None)

    return z


def assemble_zarr_with_fillvalue():
    """Create a test zarr object."""
    compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)
    z = zarr.create((10000, 10000), chunks=(1000, 1000), dtype='i1', order='C',
                    compressor=compressor)
    z._fill_value = -99

    return z


def assemble_zarr_uncompressed_2():
    """Create a test zarr object."""
    z = zarr.create((100, 100), chunks=(10, 10), dtype='i1', order='C',
                    compressor=None)

    return z


def test_as_get_orthogonal_selection():
    """Test Zarr Orthogonal selection."""
    z = assemble_zarr()
    z._cache_metadata = None
    selection = (slice(0, 2, 1), slice(4, 6, 1))
    sel = at.as_get_orthogonal_selection(z, selection=selection, out=None,
                                         fields=None)
    expected_shape = (2, 2)
    np.testing.assert_array_equal(sel.shape, expected_shape)
    expected_elem = [0, 0]
    np.testing.assert_array_equal(sel[0], expected_elem)


def test_as_get_selection():
    """Test chunk iterator."""
    z = assemble_zarr()
    selection = (slice(0, 2, 1), slice(4, 6, 1))
    indexer = OrthogonalIndexer(selection, z)
    ch = at.as_get_selection(z, indexer, out=None,
                             fields=None)
    np.testing.assert_array_equal(ch[0][0], [0, 0])
    np.testing.assert_array_equal(ch[1][0], None)
    np.testing.assert_array_equal(ch[2][0], (0, 0))


def test_as_chunk_getitem():
    """Test chunk get item."""
    z = assemble_zarr()
    z = at.make_an_array_instance_active(z)
    chunk_coords = (0, 3)
    chunk_selection = [slice(0, 2, 1)]
    out = np.array((2, 2, 2))
    out_selection = slice(0, 2, 1)
    ch = at.as_chunk_getitem(z, chunk_coords, chunk_selection, out, out_selection,
                             drop_axes=None, fields=None)
    np.testing.assert_array_equal(ch[0], (1000, 1000))
    np.testing.assert_array_equal(ch[1], [slice(0, 2, 1)])
    # PCI
    assert list(ch[2]) == [(0, 2000, (slice(0, 2, 1),))]


def test_as_chunk_getitem_with_fillvalue():
    """Test chunk get item."""
    z = assemble_zarr_with_fillvalue()
    z = at.make_an_array_instance_active(z)
    chunk_coords = (0, 3)
    chunk_selection = [slice(0, 2, 1)]
    out = np.array((2, 2, 2))
    out_selection = slice(0, 2, 1)
    ch = at.as_chunk_getitem(z, chunk_coords, chunk_selection, out, out_selection,
                             drop_axes=None, fields=None)
    np.testing.assert_array_equal(ch[0], (1000, 1000))
    np.testing.assert_array_equal(ch[1], [slice(0, 2, 1)])
    # PCI
    assert list(ch[2]) == [(0, 2000, (slice(0, 2, 1),))]


def test_process_chunk_uncompressed():
    """Test for processing chunk engine for uncompressed data"""
    z = assemble_zarr_uncompressed()
    z = at.make_an_array_instance_active(z)
    out = np.ones((1, 8))
    cdata = np.ones((1, 2))
    chunk_selection = slice(0, 1, 1)
    out_selection = np.array((0))
    ch = at.as_process_chunk(z,
                             out,
                             cdata,
                             chunk_selection,
                             drop_axes=False,
                             out_is_ndarray=True,
                             fields=None,
                             out_selection=out_selection,
                             partial_read_decode=False)

    assert is_contiguous_selection(out_selection)
    assert not is_total_slice(chunk_selection, z._chunks)
    assert out.any()
    assert out.shape == (1, 8)


def test_process_chunk_uncompressed_write_direct():
    """Test for processing chunk engine for uncompressed data"""
    z = assemble_zarr_uncompressed_2()
    z = at.make_an_array_instance_active(z)
    out = np.ones((10, 10))
    cdata = np.ones((10, 10))
    chunk_selection = (slice(0, 10, 1), slice(10, 20, 1))
    out_selection = np.array((0))
    with pytest.raises(ValueError) as exc:
        ch = at.as_process_chunk(z,
                                 out,
                                 cdata,
                                 chunk_selection,
                                 drop_axes=False,
                                 out_is_ndarray=True,
                                 fields=None,
                                 out_selection=out_selection,
                                 partial_read_decode=False)
    assert str(exc.value) == "Chunk shape (10, 80) exceeds data chunks shape (10, 10)"

    assert is_contiguous_selection(out_selection)
    assert is_total_slice(chunk_selection, z._chunks)

    z._chunks = (2, 8)
    out = np.ones((1, 8))
    cdata = np.ones((1, 2))
    chunk_selection = slice(0, 8, 1)
    out_selection = np.array((0))
    with pytest.raises(ValueError) as exc:
        ch = at.as_process_chunk(z,
                                 out,
                                 cdata,
                                 chunk_selection,
                                 drop_axes=False,
                                 out_is_ndarray=True,
                                 fields=None,
                                 out_selection=out_selection,
                                 partial_read_decode=False)
    assert str(exc.value) == "Storage chunk shape (2, 8) exceeds permitted output data shape (1, 8)."


def test_process_chunk_uncompressed_with_compressor():
    """Test for processing chunk engine for uncompressed data"""
    z = assemble_zarr()
    z = at.make_an_array_instance_active(z)
    out = np.ones((1, 8))
    cdata = np.ones((1, 2))
    chunk_selection = slice(0, 1, 1)
    out_selection = np.array((0))
    raised = f'cdata {cdata} is an ndarray, can not decompress.'
    with pytest.raises(TypeError) as exc:
        ch = at.as_process_chunk(z,
                                 out,
                                 cdata,
                                 chunk_selection,
                                 drop_axes=False,
                                 out_is_ndarray=True,
                                 fields=None,
                                 out_selection=out_selection,
                                 partial_read_decode=False)
    assert str(exc.value) == raised
