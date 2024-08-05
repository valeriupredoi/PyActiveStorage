import os
import pytest

import numpy as np
from netCDF4 import Dataset
from pathlib import Path
import s3fs

from activestorage.active import Active
from activestorage.config import *


def mytest():
    """
    Test again a native model, this time around netCDF4 loadable with h5py
    Also, this has _FillValue and missing_value
    """
    ncfile, v = "cesm2_native.nc","TREFHT"
    ncfile, v  = "CMIP6-test.nc", 'tas'
    #ncfile, v = "chunked.hdf5", "dataset1"
    #ncfile, v = 'daily_data.nc', 'ta'
    mypath = Path(__file__).parent
    uri = str(mypath/ncfile)
    results = []
    for av in [0,1,2]:

        active = Active(uri, v, None)
        active._version = av
        if av > 0:
            active.method="mean"
        if v == "dataset1":
            d = active[2,:]
        else:
            d = active[4:5, 1:2]
        print(active.metric_data)
        if av == 0:
            d = np.mean(d)
        results.append(d)


    # check for active
    np.testing.assert_allclose(results[0],results[1], rtol=1e-6)
    np.testing.assert_allclose(results[1],results[2], rtol=1e-6)



if __name__=="__main__":
    mytest()