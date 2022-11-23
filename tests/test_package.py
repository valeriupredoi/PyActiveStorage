import activestorage

from activestorage import Active as act
from activestorage.active import _read_config_file as read_conf


# test version
def test_version():
    assert hasattr(activestorage, "__version__")
    assert activestorage.__version__ == "0.0.1"
    print(activestorage.__version__)

# check activestorage class
def test_activestorage_class_attrs():
    assert hasattr(activestorage, "Active")
    assert hasattr(activestorage, "active")
    assert hasattr(activestorage, "storage")
    assert hasattr(activestorage, "netcdf_to_zarr")

# check Active class
def test_active_class_attrs():
    assert hasattr(act, "__new__")
    assert hasattr(act, "__getitem__")
    assert hasattr(act, "__init__")
    assert hasattr(act, "_from_storage")
    assert hasattr(act, "_get_active")
    assert hasattr(act, "_get_selection")
    assert hasattr(act, "_process_chunk")
    assert hasattr(act, "_via_kerchunk")
    assert hasattr(act, "components")
    assert hasattr(act, "method")
    assert hasattr(act, "ncvar")


# check validity of conf files
def test_read_config_file():
    """Test validity of package-level files."""
    posix_mandatory_keys = ["version", "methods"]
    s3_mandatory_keys = ["version", "methods"]
    posix_file = read_conf("Posix")
    s3_file = read_conf("S3")
    print(posix_file)
    print(s3_file)
    for mandatory_key in posix_mandatory_keys:
        assert mandatory_key in posix_file
    for mandatory_key in s3_mandatory_keys:
        assert mandatory_key in s3_file
    
