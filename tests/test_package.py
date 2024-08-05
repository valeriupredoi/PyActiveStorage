import activestorage

from activestorage import Active as act


# test version
def test_version():
    assert hasattr(activestorage, "__version__")
    assert activestorage.__version__ == "0.0.2"
    print(activestorage.__version__)

# check activestorage class
def test_activestorage_class_attrs():
    assert hasattr(activestorage, "Active")
    assert hasattr(activestorage, "active")
    assert hasattr(activestorage, "storage")

# check Active class
def test_active_class_attrs():
    assert hasattr(act, "__new__")
    assert hasattr(act, "__getitem__")
    assert hasattr(act, "__init__")
    assert hasattr(act, "_from_storage")
    assert hasattr(act, "_get_active")
    assert hasattr(act, "_get_selection")
    assert hasattr(act, "_process_chunk")
    assert hasattr(act, "components")
    assert hasattr(act, "method")
    assert hasattr(act, "ncvar")
