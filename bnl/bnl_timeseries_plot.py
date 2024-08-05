from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import os


plt.rcParams["figure.figsize"] = (8,12)

data = {'timeseries1':'Hm1, Active: 5 MB Blks',
        'timeseries2':'Hm1, Active: 1 MB Blks',
        'timeseries2v1':'Hm1, Local: 1 MB Blks',
        'timeseries2-uor-t1-a':'Uni, Active: T1, 1 MB Blks',
        'timeseries2-uor-t1-b':'Uni, Active: T1, 1 MB Blks',
        'timeseries2-uor-t1-c':'Uni, Active: T1, 1 MB Blks',
        'timeseries2-uor-t100-a':'Uni, Active: T100, 1 MB Blks',
        'timeseries2-uor-t100-b':'Uni, Active: T100, 1 MB Blks',
        'timeseries1-uor-t100-a':'Uni, Local: T100, 1 MB Blks',
        'timeseries1-uor-t100-b':'Uni, Local: T100, 1 MB Blks',
        'timeseries2v1-uor-t1':'Uni, Local: T1, 1 MB Blks',
        'timeseries1-uor-t1':'Uni, Active: T1, 5 MB Blks',
                }
tt =  []

mypath = Path(__file__).parent

logfiles = mypath.glob('timeseries3-H*.log')

#for d,t in data.items():

for ff in logfiles:
      
    #fname1 = mypath/f'{d}.log'
    #fname2 = mypath/f'{d}.metrics.txt'
    #os.system(f'grep -a  "dataset shape" {fname1} > {fname2}')
    #with open(fname2,'r') as f:
    with open(ff,'r') as f:
        lines = f.readlines()
        dicts = [eval(l) for l in lines if l.startswith('{') ]
        nct = dicts[0]['load nc time']
        rt = [v['reduction time (s)'] for v in dicts]
        summary = lines[-1][9:]
        overall_time = float(summary.split(',')[-1][:-2])
        tt.append((overall_time, rt, summary))
    #os.system(f'rm {fname2}')

tts = sorted(tt)
curve = {k:v for i,v,k in tts}

for k in curve.keys():

    if 'Local' in k: 
        ls = 'dashed'
    else:
        ls = 'solid'

    plt.plot(curve[k], label=k, linestyle=ls)


plt.legend(bbox_to_anchor=(0.7, -0.1), fontsize=8)
plt.title('Comparison of reduction parameters')
plt.ylabel('Time to reduce each timestep (s)')
plt.tight_layout()
plt.savefig('home.png')
plt.show()



