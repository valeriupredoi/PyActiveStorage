import os

import netCDF4
import s3fs
import activestorage.config
import tempfile

from activestorage.dummy_data import make_vanilla_ncdata


# Force True for S3 exploratory tests
USE_S3 = True

# URL of Reductionist S3 Active Storage server.
S3_ACTIVE_STORAGE_URL = "http://localhost:8080"

# URL of S3 object store.
S3_URL = "http://localhost:9000"

# S3 access key / username.
S3_ACCESS_KEY = "minioadmin"

# S3 secret key / password.
S3_SECRET_KEY = "minioadmin"

# S3 bucket.
S3_BUCKET = "pyactivestorage"

def make_tempfile():
    """Make dummy data."""
    temp_folder = tempfile.mkdtemp()
    s3_testfile = os.path.join(temp_folder,
                               's3_test_bizarre.nc')
    print(f"S3 Test file is {s3_testfile}")
    if not os.path.exists(s3_testfile):
        make_vanilla_ncdata(filename=s3_testfile)

    local_testfile = os.path.join(temp_folder,
                                  'local_test_bizarre.nc')
    print(f"Local Test file is {local_testfile}")
    if not os.path.exists(local_testfile):
        make_vanilla_ncdata(filename=local_testfile)

    return s3_testfile, local_testfile


def upload_to_s3(server, username, password, bucket, object, rfile):
    """Upload a file to an S3 object store."""
    s3_fs = s3fs.S3FileSystem(key=username, secret=password, client_kwargs={'endpoint_url': server})
    # Make sure s3 bucket exists
    try:
        s3_fs.mkdir(bucket)
    except FileExistsError:
        pass

    s3_fs.put_file(rfile, os.path.join(bucket, object))

    return os.path.join(bucket, object)


def test_load_s3_file():
    s3_testfile, local_testfile = make_tempfile()

    # put s3 dummy data onto S3. then rm from local
    object = os.path.basename(s3_testfile)
    bucket_file = upload_to_s3(S3_URL, S3_ACCESS_KEY, S3_SECRET_KEY,
                               S3_BUCKET, object, s3_testfile)
    os.remove(s3_testfile)
    s3_testfile_uri = os.path.join("s3://", bucket_file)

    fs = s3fs.S3FileSystem(key=S3_ACCESS_KEY,  # eg "minioadmin" for Minio
                           secret=S3_SECRET_KEY,  # eg "minioadmin" for Minio
                           client_kwargs={'endpoint_url': S3_URL})  # eg "http://localhost:9000" for Minio

    print(f"S3 file URI: {s3_testfile_uri}")
    print("Will attempt to ncdump for now")
    # not clear which is the loading procedure just yet
    # one option is via s3fs straight to Dataset - not working
    # with fs.open(s3_testfile_uri + '#mode=bytes', 'rb') as s3file:
    # ds = netCDF4.Dataset(s3_testfile_uri + '#mode=bytes', 'r')

    # another option is to get bytes in memory
    # this works but blows up the entire data into memory
    with fs.open(s3_testfile_uri, 'rb') as s3f:
        print(s3f.metadata)
        nc_bytes = s3f.read()
    ds = netCDF4.Dataset(f'inmemory.nc', memory=nc_bytes)
    print(f"netCDF4.Dataset loaded from memory: {ds}")
    print("netCDF4.Dataset variable loaded from memory: ", ds["data"])
    print("check if data", ds["data"][:])

    # see what diskless do
    with fs.open(s3_testfile_uri, 'rb') as s3f:
        ds2 = netCDF4.Dataset(s3f, diskless=True)
        print(ds2)

    print(x)


# def test_s3_load_via_netcdf4():
#     ds = load_s3_file()
#     print(ds)
