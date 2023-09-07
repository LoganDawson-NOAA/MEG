import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits
mpl_toolkits.__path__.append('/gpfs/dell2/emc/modeling/noscrub/gwv/py/lib/python/basemap-1.2.1-py3.6-linux-x86_64.egg/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import colors as c
#from pylab import *
import numpy as np
import pygrib, datetime, time, os, sys, subprocess
import multiprocessing, itertools, collections
import scipy, ncepy
from ncepgrib2 import Grib2Encode, Grib2Decode
import grib2io
import dawsonpy



#Determine initial date/time
try:
   cycle = str(sys.argv[1])
except IndexError:
   cycle = None

if cycle is None:
   cycle = raw_input('Enter initial time (YYYYMMDDHH): ')

YYYY = int(cycle[0:4])
MM   = int(cycle[4:6])
DD   = int(cycle[6:8])
HH   = int(cycle[8:10])
print(YYYY, MM, DD, HH)

date_str = datetime.datetime(YYYY,MM,DD,HH,0,0)


# Determine desired model
try:
   model_str = str(sys.argv[2])
except IndexError:
   model_str = None

if model_str is None:
   print('Model string options: GFS, GEFS, FV3, EC, NAM, NAMNEST, RAP, HRRR, HRRRX, HIRESW, or HREF')
   model_str = raw_input('Enter desired model: ')
   ## GEFS options: GEFS, GEFSMEAN, GEFSSPREAD, GEFSCTRL, GEFSMEMS
   ## HREF options: HREF, HREFPROB, HREFMEAN, HREFPMMN, HREFAVRG
   ## HIRESW options: HIRESW, HIRESWARW, HIRESWARW2, HIRESWNMMB

fhrs = [0]

if model_str == 'HRRR':
    fil = '/gpfs/hps/nco/ops/com/hrrr/prod/hrrr.'+cycle[0:8]+'/conus/'+str.lower(model_str[0:6])+'.t'+cycle[8:10]+'z.wrfprsf'+str(fhrs[0]).zfill(2)+'.grib2'
elif model_str == 'GFS':
    fil = '/gpfs/dell1/nco/ops/com/gfs/prod/gfs.'+cycle[0:8]+'/'+cycle[8:10]+'/'+ \
          'atmos/'+str.lower(model_str)+'.t'+cycle[8:10]+'z.pgrb2.0p25.f'+str(fhrs[0]).zfill(3)

grbs = grib2io.open(fil,mode='r')

msgs = grbs['CAPE']


if model_str == 'HRRR':
    grb = grbs.select(shortName='MSLMA')[0]
elif model_str == 'GFS':
    grb = grbs.select(shortName='MSLET')[0]

mslp = grb.data()*0.01

lats, lons = grb.latlons()

print('Lat/lon read from',grb.fullName)

grbs.close()

