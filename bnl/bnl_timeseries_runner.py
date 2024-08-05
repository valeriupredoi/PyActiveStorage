from bnl_timeseries import timeseries
import sys

location = 'Hm1'
blocks = [1,5]
version = [1,2]
threads = [1,100]
iterations = ['a','b','c']

for b in blocks:
    for v in version:
        for t in threads:
            for i in iterations:
                with open(f'timeseries3-{location}-{b}-{v}-{t}-{i}.log','w') as sys.stdout:
                    timeseries(location, b, v, t)
