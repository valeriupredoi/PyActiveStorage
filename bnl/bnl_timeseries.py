# Simple code to get hemispheric mean of the lowest level of air temperature from 
# float UM_m01s30i204_vn1106(time, air_pressure_3, latitude_1, longitude_1) ;
#		UM_m01s30i204_vn1106:long_name = "TEMPERATURE ON P LEV/UV GRID" ;" ;
#		UM_m01s30i204_vn1106:units = "K" ;
#		UM_m01s30i204_vn1106:cell_methods = "time: point" ;
# with
# 	time = 40 ;
#	latitude_1 = 1921 ;
#	latitude = 1920 ;
#	longitude_1 = 2560 ;
#	air_pressure_3 = 11 ;

#  This variable is held in an 18 GB file on S3 storage

from activestorage.active import Active
import numpy as np
from time import time

S3_BUCKET = "bnl"
S3_URL = "https://uor-aces-o.s3-ext.jc.rl.ac.uk"
ACTIVE_URL = "https://192.171.169.248:8080"

def timeseries(location='uni', blocks_MB=1, version=2, threads=100):

    invoke = time()
    storage_options = {
        'key': "f2d55c6dcfc7618b2c34e00b58df3cef",
        'secret': "$/'#M{0{/4rVhp%n^(XeX$q@y#&(NM3W1->~N.Q6VP.5[@bLpi='nt]AfH)>78pT",
        'client_kwargs': {'endpoint_url': f"{S3_URL}"}, 
            'default_fill_cache':False,
            'default_cache_type':"readahead",
            'default_block_size': blocks_MB * 2**20
    }

    filename = 'ch330a.pc19790301-def.nc'
    uri = S3_BUCKET + '/' + filename
    var = "UM_m01s30i204_vn1106"

    active = Active(uri, var, storage_type="s3", max_threads=threads,
                            storage_options=storage_options,
                            active_storage_url=ACTIVE_URL)

    # set active to use the remote reductionist.
    # (we intend to change the invocation method
    # to something more obvious and transparent.)
    active._version = version

    # and set the operation, again, the API will change
    active._method = "mean"

    # get hemispheric mean timeseries:
    # (this would be more elegant in cf-python)
    ts = []
    md = []
    for i in range(40):
        ts.append(active[i,0,0:960,:][0])
        # get some performance diagnostics from pyactive
        print(active.metric_data)
        if i == 0:
            nct = active.metric_data['load nc time']
        md.append(active.metric_data['reduction time (s)'])

    result = np.array(ts)
    print(result)
    complete = time()
    method = {1:'Local',2:'Active'}[version]
    titlestring = f"{location}:{method} (T{threads},BS{blocks_MB}): {nct:.3}s,{sum(md):.4}s,{complete-invoke:.4}s"
    print('Summary: ',titlestring)
