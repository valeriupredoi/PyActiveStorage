"""
Module to hold Zarr lift-up code.

We're effectively subclassing zarr.core.Array, but not actually doing so, 
instead we're providing tools to hack instances of it
"""
import numpy as np
from zarr.core import Array

# import other zarr gubbins used in the methods we override
from zarr.indexing import (
    OrthogonalIndexer,
    PartialChunkIterator,
    check_fields,
    is_contiguous_selection,

)
from zarr.util import (
    check_array_shape,
    is_total_slice,
    PartialReadBuffer,
)

from numcodecs.compat import ensure_ndarray
from zarr.errors import ArrayIndexError


def make_an_array_instance_active(instance):
    """ 
    Given a zarr array instance, override some key methods so 
    we can do active storage things. Note this only works for
    normal and _ methods and would not work on __ methods.
    
    This an ugly hack for development to avoid having to hack
    zarr internal in a fork.
    """

    instance.get_orthogonal_selection = as_get_orthogonal_selection.__get__(instance, Array)
    instance._get_selection = as_get_selection.__get__(instance, Array)
    instance._chunk_getitem = as_chunk_getitem.__get__(instance, Array)
    instance._process_chunk = as_process_chunk.__get__(instance, Array)
    instance._process_chunk_V = as_process_chunk_V.__get__(instance, Array)

    return instance

def as_get_orthogonal_selection(self, selection, out=None,
                                 fields=None):
    """
    Retrieve data by making a selection for each dimension of the array. For
    example, if an array has 2 dimensions, allows selecting specific rows and/or
    columns. The selection for each dimension can be either an integer (indexing a
    single item), a slice, an array of integers, or a Boolean array where True
    values indicate a selection.
    Parameters
    ----------
    selection : tuple
        A selection for each dimension of the array. May be any combination of int,
        slice, integer array or Boolean array.
    out : ndarray, optional
        If given, load the selected data directly into this array.
    fields : str or sequence of str, optional
        For arrays with a structured dtype, one or more fields can be specified to
        extract data for.
    Returns
    -------
    out : ndarray
        A NumPy array containing the data for the requested selection.
    Examples
    --------
    Setup a 2-dimensional array::
        >>> import zarr
        >>> import numpy as np
        >>> z = zarr.array(np.arange(100).reshape(10, 10))
    Retrieve rows and columns via any combination of int, slice, integer array and/or
    Boolean array::
        >>> z.get_orthogonal_selection(([1, 4], slice(None)))
        array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
        >>> z.get_orthogonal_selection((slice(None), [1, 4]))
        array([[ 1,  4],
                [11, 14],
                [21, 24],
                [31, 34],
                [41, 44],
                [51, 54],
                [61, 64],
                [71, 74],
                [81, 84],
                [91, 94]])
        >>> z.get_orthogonal_selection(([1, 4], [1, 4]))
        array([[11, 14],
                [41, 44]])
        >>> sel = np.zeros(z.shape[0], dtype=bool)
        >>> sel[1] = True
        >>> sel[4] = True
        >>> z.get_orthogonal_selection((sel, sel))
        array([[11, 14],
                [41, 44]])
    For convenience, the orthogonal selection functionality is also available via the
    `oindex` property, e.g.::
        >>> z.oindex[[1, 4], :]
        array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
        >>> z.oindex[:, [1, 4]]
        array([[ 1,  4],
                [11, 14],
                [21, 24],
                [31, 34],
                [41, 44],
                [51, 54],
                [61, 64],
                [71, 74],
                [81, 84],
                [91, 94]])
        >>> z.oindex[[1, 4], [1, 4]]
        array([[11, 14],
                [41, 44]])
        >>> sel = np.zeros(z.shape[0], dtype=bool)
        >>> sel[1] = True
        >>> sel[4] = True
        >>> z.oindex[sel, sel]
        array([[11, 14],
                [41, 44]])
    Notes
    -----
    Orthogonal indexing is also known as outer indexing.
    Slices with step > 1 are supported, but slices with negative step are not.
    See Also
    --------
    get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
    get_coordinate_selection, set_coordinate_selection, set_orthogonal_selection,
    vindex, oindex, __getitem__, __setitem__
    """

    # refresh metadata
    if not self._cache_metadata:
        self._load_metadata()

    # check args
    check_fields(fields, self._dtype)

    # setup indexer
    indexer = OrthogonalIndexer(selection, self)

    return self._get_selection(indexer=indexer, out=out,
                                fields=fields)


def as_get_selection(self, indexer, out=None,
                    fields=None):

    # We iterate over all chunks which overlap the selection and thus contain data
    # that needs to be extracted. Each chunk is processed in turn, extracting the
    # necessary data and storing into the correct location in the output array.

    # N.B., it is an important optimisation that we only visit chunks which overlap
    # the selection. This minimises the number of iterations in the main for loop.

    # check fields are sensible
    out_dtype = check_fields(fields, self._dtype)

    # determine output shape
    out_shape = indexer.shape

    # setup output array
    if out is None:
        out = np.empty(out_shape, dtype=out_dtype, order=self._order)
    else:
        check_array_shape('out', out, out_shape)

    chunks_info = []
    chunks_locs = []

    # iterate over chunks
    if not hasattr(self.chunk_store, "getitems") or \
        any(map(lambda x: x == 0, self.shape)):
        # sequentially get one key at a time from storage
        for chunk_coords, chunk_selection, out_selection in indexer:

            # load chunk selection into output array
            pci = self._chunk_getitem(chunk_coords, chunk_selection, out, out_selection,
                                        drop_axes=indexer.drop_axes, fields=fields)
                                        
            chunks_info.append(pci)
            chunks_locs.append(chunk_coords)
    else:
        # allow storage to get multiple items at once
        lchunk_coords, lchunk_selection, lout_selection = zip(*indexer)
        self._chunk_getitems(lchunk_coords, lchunk_selection, out, lout_selection,
                                drop_axes=indexer.drop_axes, fields=fields)

    if out.shape:
        return out, chunks_info, chunks_locs
    else:
        return out[()], chunks_info, chunks_locs


def as_chunk_getitem(self, chunk_coords, chunk_selection, out, out_selection,
                    drop_axes=None, fields=None):
    """Obtain part or whole of a chunk.
    Parameters
    ----------
    chunk_coords : tuple of ints
        Indices of the chunk.
    chunk_selection : selection
        Location of region within the chunk to extract.
    out : ndarray
        Array to store result in.
    out_selection : selection
        Location of region within output array to store results in.
    drop_axes : tuple of ints
        Axes to squeeze out of the chunk.
    fields
        TODO
    """
    out_is_ndarray = True
    try:
        out = ensure_ndarray(out)
    except TypeError:
        out_is_ndarray = False

    assert len(chunk_coords) == len(self._cdata_shape)

    try:
        pci_info = self._process_chunk_V(chunk_selection)
    except KeyError:
        # chunk not initialized
        if self._fill_value is not None:
            if fields:
                fill_value = self._fill_value[fields]
            else:
                fill_value = self._fill_value
            out[out_selection] = fill_value
        pci_info = self._process_chunk_V(chunk_selection)

    return pci_info


def as_process_chunk(
    self,
    out,
    cdata,
    chunk_selection,
    drop_axes,
    out_is_ndarray,
    fields,
    out_selection,
    partial_read_decode=False,
):
    """Take binary data from storage and fill output array"""
    if (out_is_ndarray and
            not fields and
            is_contiguous_selection(out_selection) and
            is_total_slice(chunk_selection, self._chunks) and
            not self._filters and
            self._dtype != object):

        dest = out[out_selection]
        write_direct = (
            dest.flags.writeable and
            (
                (self._order == 'C' and dest.flags.c_contiguous) or
                (self._order == 'F' and dest.flags.f_contiguous)
            )
        )

        if write_direct:

            # optimization: we want the whole chunk, and the destination is
            # contiguous, so we can decompress directly from the chunk
            # into the destination array
            if self._compressor:
                if isinstance(cdata, PartialReadBuffer):
                    cdata = cdata.read_full()
                self._compressor.decode(cdata, dest)
            else:
                chunk = ensure_ndarray(cdata).view(self._dtype)
                if np.prod(chunk.shape) > np.prod(self._chunks):
                    raise ValueError(f"Chunk shape {chunk.shape} exceeds "
                                     f"data chunks shape {self._chunks}")
                chunk = chunk.reshape(self._chunks, order=self._order)
                np.copyto(dest, chunk)
            return

    # decode chunk
    try:
        if partial_read_decode:
            cdata.prepare_chunk()
            # size of chunk
            tmp = np.empty(self._chunks, dtype=self.dtype)
            index_selection = PartialChunkIterator(chunk_selection, self.chunks)
            for start, nitems, partial_out_selection in index_selection:
                expected_shape = [
                    len(
                        range(*partial_out_selection[i].indices(self.chunks[0] + 1))
                    )
                    if i < len(partial_out_selection)
                    else dim
                    for i, dim in enumerate(self.chunks)
                ]
                cdata.read_part(start, nitems)
                chunk_partial = self._decode_chunk(
                    cdata.buff,
                    start=start,
                    nitems=nitems,
                    expected_shape=expected_shape,
                )
                tmp[partial_out_selection] = chunk_partial
            out[out_selection] = tmp[chunk_selection]
            return
    except ArrayIndexError:
        cdata = cdata.read_full()
    if self._compressor and isinstance(cdata, np.ndarray):
        raise TypeError(f"cdata {cdata} is an ndarray, can not decompress.")
    chunk = self._decode_chunk(cdata)

    # select data from chunk
    if fields:
        chunk = chunk[fields]
    tmp = chunk[chunk_selection]
    if drop_axes:
        tmp = np.squeeze(tmp, axis=drop_axes)

    # store selected data in output
    if np.prod(tmp.shape) > np.prod(out.shape):
        raise ValueError(f"Storage chunk shape {tmp.shape} exceeds permitted "
                         f"output data shape {out.shape}.")
    out[out_selection] = tmp

def as_process_chunk_V(self, chunk_selection):
    """Run an instance of PCI inside the engine."""
    index_selection = PartialChunkIterator(chunk_selection, self.chunks)
    for _, _, _ in index_selection:
        return self.chunks, chunk_selection, index_selection
