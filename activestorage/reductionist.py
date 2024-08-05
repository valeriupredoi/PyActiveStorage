"""Reductionist S3 Active Storage server storage interface module."""

import collections.abc
import http.client
import json
import requests
import numcodecs
import numpy as np
import sys
import typing

DEBUG = 0


def get_session(username: str, password: str, cacert: typing.Optional[str]) -> requests.Session:
    """Create and return a client session object.

    :param username: S3 username / access key
    :param password: S3 password / secret key
    :returns: a client session object.
    """
    session = requests.Session()
    session.auth = (username, password)
    session.verify = cacert or False
    return session


def reduce_chunk(session, server, source, bucket, object,
                 offset, size, compression, filters, missing, dtype, shape,
                 order, chunk_selection, operation):
    """Perform a reduction on a chunk using Reductionist.

    :param server: Reductionist server URL
    :param cacert: Reductionist CA certificate path
    :param source: S3 URL
    :param bucket: S3 bucket
    :param object: S3 object
    :param offset: offset of data in object
    :param size: size of data in object
    :param compression: optional `numcodecs.abc.Codec` compression codec
    :param filters: optional list of `numcodecs.abc.Codec` filter codecs
    :param missing: optional 4-tuple describing missing data
    :param dtype: numpy data type
    :param shape: will be a tuple, something like (3,3,1), this is the
                  dimensionality of the chunk itself
    :param order: typically 'C' for c-type ordering
    :param chunk_selection: N-tuple where N is the length of `shape`, and each
                            item is an integer or slice.  e.g.  (slice(0, 2,
                            1), slice(1, 3, 1), slice(0, 1, 1))
                            this defines the part of the chunk which is to be
                            obtained or operated upon.
    :param operation: name of operation to perform
    :returns: the reduced data as a numpy array or scalar
    :raises ReductionistError: if the request to Reductionist fails
    """

    request_data = build_request_data(source, bucket, object, offset, size, compression, filters, missing, dtype, shape, order, chunk_selection)
    if DEBUG:
        print(f"Reductionist request data dictionary: {request_data}")
    api_operation = "sum" if operation == "mean" else operation or "select"
    url = f'{server}/v1/{api_operation}/'
    response = request(session, url, request_data)

    if response.ok:
        return decode_result(response)
    else:
        decode_and_raise_error(response)


def encode_byte_order(dtype):
    """Encode the byte order (endianness) of a dtype in a JSON-compatible format."""
    if dtype.byteorder == '=':
        return sys.byteorder
    elif dtype.byteorder == '<':
        return 'little'
    elif dtype.byteorder == '>':
        return 'big'
    assert False, "Unexpected byte order {dtype.byteorder}"


def encode_selection(selection):
    """Encode a chunk selection in a JSON-compatible format."""
    def encode_slice(s):
        if isinstance(s, slice):
            return [s.start, s.stop, s.step]
        else:
            # Integer - select single value
            return [s, s + 1, 1]

    return [encode_slice(s) for s in selection]


def encode_filter(filter):
    """Encode a filter algorithm in a JSON-compatible format."""
    if filter.codec_id == "shuffle":
        return {"id": filter.codec_id, "element_size": filter.elementsize}
    else:
        raise ValueError(f"Unsupported filter {filter})")


def encode_filters(filters):
    """Encode the list of filter algorithms in a JSON-compatible format."""
    return [encode_filter(filter) for filter in filters or []]


def encode_dvalue(value):
    """Encode a data value in a JSON-compatible format."""
    if isinstance(value, np.float32):
        # numpy cannot encode float32, so convert it to a float64.
        return np.float64(value)
    return value


def encode_missing(missing):
    """Encode missing data in a JSON-compatible format."""
    fill_value, missing_value, valid_min, valid_max = missing
    # fill_value and missing_value are effectively the same when reading data.
    missing_value = fill_value or missing_value
    if missing_value:
        if isinstance(missing_value, collections.abc.Sequence):
            return {"missing_values": [encode_dvalue(v) for v in missing_value]}
        elif isinstance(missing_value, np.ndarray):
            return {"missing_values": [encode_dvalue(v) for v in missing_value]}
        else:
            return {"missing_value": encode_dvalue(missing_value)}
    if valid_min and valid_max:
        return {"valid_range": [encode_dvalue(valid_min), encode_dvalue(valid_max)]}
    if valid_min:
        return {"valid_min": encode_dvalue(valid_min)}
    if valid_max:
        return {"valid_max": encode_dvalue(valid_max)}
    assert False, "Expected missing values not found"


def build_request_data(source: str, bucket: str, object: str, offset: int,
                       size: int, compression, filters, missing, dtype, shape,
                       order, selection) -> dict:
    """Build request data for Reductionist API."""
    request_data = {
        'source': source,
        'bucket': bucket,
        'object': object,
        'dtype': dtype.name,
        'byte_order': encode_byte_order(dtype),
        'offset': int(offset),
        'size': int(size),
        'order': order,
    }
    if shape:
        request_data["shape"] = shape
    if selection:
        request_data["selection"] = encode_selection(selection)
    if compression:
        # Currently the numcodecs codec IDs map to the supported compression
        # algorithm names in the active storage server. If that changes we may
        # need to maintain a mapping here.
        request_data["compression"] = {"id": compression.codec_id}
    if filters:
        request_data["filters"] = encode_filters(filters)
    if any(missing):
        request_data["missing"] = encode_missing(missing)

    return {k: v for k, v in request_data.items() if v is not None}


def request(session: requests.Session, url: str, request_data: dict):
    """Make a request to a Reductionist API."""
    response = session.post(
        url,
        json=request_data,
    )
    return response


def decode_result(response):
    """Decode a successful response, return as a 2-tuple of (numpy array or scalar, count)."""
    dtype = response.headers['x-activestorage-dtype']
    shape = json.loads(response.headers['x-activestorage-shape'])
    result = np.frombuffer(response.content, dtype=dtype)
    result = result.reshape(shape)
    count = json.loads(response.headers['x-activestorage-count'])
    return result, count


class ReductionistError(Exception):
    """Exception for Reductionist failures."""

    def __init__(self, status_code, error):
        super(ReductionistError, self).__init__(f"Reductionist error: HTTP {status_code}: {error}")


def decode_and_raise_error(response):
    """Decode an error response and raise ReductionistError."""
    try:
        error = json.dumps(response.json())
        raise ReductionistError(response.status_code, error)
    except requests.exceptions.JSONDecodeError as exc:
        raise ReductionistError(response.status_code, "-") from exc
