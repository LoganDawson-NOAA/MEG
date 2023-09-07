#!/usr/bin/env python

import numpy as np
import datetime, time, os, sys, subprocess

DIR = os.getcwd()

cycle = str(sys.argv[1])
YYYY = int(cycle[0:4])
MM   = int(cycle[4:6])
DD   = int(cycle[6:8])
HH   = int(cycle[8:10])
date_str = datetime.datetime(YYYY,MM,DD,HH,0,0)

fhrb = int(sys.argv[2])
fhre = int(sys.argv[3])
step = int(sys.argv[4])
fhrs = np.arange(fhrb,fhre+1,step)

valid_list = [date_str + datetime.timedelta(hours=int(x)) for x in fhrs]

f = open(DIR+'/'+cycle+'_valids.txt',"w+")

for k in range(len(valid_list)):
   f.write(valid_list[k].strftime("%Y%m%d%H")+" \n")

f.close()


