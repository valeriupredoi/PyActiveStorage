
from activestorage.active import Active
import os
from pathlib import Path
from netCDF4 import Dataset
import numpy as np
import pyfive
import s3fs

from activestorage.active import load_from_s3

S3_BUCKET = "bnl"

def simple(filename, var):

    S3_URL = 'https://uor-aces-o.s3-ext.jc.rl.ac.uk/'
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': S3_URL})
    uri = 'bnl/'+filename

    with fs.open(uri,'rb') as s3file2:
        f2 = pyfive.File(s3file2)
        print(f2[var])

    with fs.open(uri, 'rb') as s3file:
        ds = pyfive.File(s3file)
        print(ds[var])

    #f2 = load_from_s3(uri, storage_options={'client_kwargs':{"endpoint_url":S3_URL}})
      
def ex_test(ncfile, var):
    """
    Test use of datasets with compression and filters applied for a real
    CMIP6 dataset (CMIP6-test.nc) - an IPSL file.

    This is for a special anon=True bucket connected to via valid key.secret
    """
    storage_options = {
        'key': "f2d55c6dcfc7618b2c34e00b58df3cef",
        'secret': "$/'#M{0{/4rVhp%n^(XeX$q@y#&(NM3W1->~N.Q6VP.5[@bLpi='nt]AfH)>78pT",
        'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"}
    }
    active_storage_url = "https://192.171.169.248:8080"


    mypath = Path(__file__).parent
    uri = str(mypath/ncfile)
    with Dataset(uri) as nc_data:
        nc_min = np.min(nc_data[var][0:2,4:6,7:9])
    print(f"Numpy min from compressed file {nc_min}")

    ofile = os.path.basename(uri)
    test_file_uri = os.path.join(
        S3_BUCKET,
        ofile
    )
    print("S3 Test file path:", test_file_uri)

    for av in [0,1,2]:

        active = Active(test_file_uri, var, storage_type="s3",
                        storage_options=storage_options,
                        active_storage_url=active_storage_url)

        active._version = av
        if av > 0: 
            active._method = "min"

        result = active[0:2,4:6,7:9]
        print(active.metric_data)
        if av == 0:
            result = np.min(result)
        assert nc_min == result
        assert result == 239.25946044921875



if __name__=="__main__":
    ncfile, var = 'CMIP6-test.nc','tas'
    #ncfile, var = 'test_partially_missing_data.nc','data'
    simple(ncfile, var)
    ex_test(ncfile, var)