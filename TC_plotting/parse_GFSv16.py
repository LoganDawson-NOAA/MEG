# Author: L Dawson
#
# Script to pull fcst files from HPSS, rename, and save in desired data directory
# Desired cycle and model string can be passed in from command line
# If no arguments are passed in, script will prompt user for these inputs
#
# Run as:
# python parse_adeck.py $MODEL $TC_name/ID
# python parse_adeck.py GFS FlorenceAL06
#
# Script History Log:
# 2018-10-03 L Dawson initial version to pull HWRF/HMON ATCF files from tape


import numpy as np
import datetime, os, sys, subprocess
import re, csv, glob


# Determine desired model
try:
   model_str = str(sys.argv[1])
except IndexError:
   model_str = None

if model_str is None:
   print('Model string options: GFS, AVNO, EC, ECMWF, EMX, UKMet, EGRR, CMC, HWRF, HMON, NAM')
   print('Model string options (early): AVNI, CMCI, HWFI, HMNI')
   print('Model string options (ensemble mean): GEFSMean, ECENSmean')
   model_str = raw_input('Enter desired model: ')


if str.upper(model_str) == 'GFS':
   model = 'AVNO'
elif str.upper(model_str) == 'EC' or str.upper(model_str) == 'ECMWF':
   model = 'ECMO'
elif str.upper(model_str) == 'UKMET':
   model = 'EGRR'
elif str.upper(model_str) == 'GEFSMEAN':
   model = 'AEMN'
elif str.upper(model_str) == 'ECENSMEAN':
   model = 'EEMN'
else:
   model = model_str


# Get TC name and number
try:
   TC = str(sys.argv[2])
except IndexError:
   TC = None

if TC is None:
   print('Enter TC name, number, and year as one string')
   print('Example: FlorenceAL062018')
   TC = input('Enter TC name/number/year: ')

TC_name = TC[:-8]
TC_number = TC[-8:-4]
YYYY = TC[-4:]
print(TC_name, TC_number, YYYY)




# Set path and create data directory (if not already created)

DIR = os.getcwd()

ADECK_DIR = '/gpfs/dell2/nhc/noscrub/data/atcf-noaa/aid_nws'
GFS_DIR = '/gpfs/dell6/emc/modeling/noscrub/emc.glopara/archive/gfsv16'
DATA_DIR = os.path.join('/gpfs/dell2/emc/verification/noscrub/Logan.Dawson/MEG/', TC_name, 'data')

if not os.path.exists(DATA_DIR):
      os.makedirs(DATA_DIR)


cycles=[]

# Get list of cycles based on matching files in DATA_DIR
#filelist = [f for f in glob.glob(DATA_DIR+'/atcfunix.gfs.*')]
filelist = [f for f in glob.glob(GFS_DIR+'/trak.gfso.atcfunix.altg.202008*')]


cycles_unordered = [filelist[x][-10:] for x in range(len(filelist))]
cycles_int = [int(x) for x in cycles_unordered]
cycles_int.sort()
cycles = [str(x) for x in cycles_int]


for cycle in cycles:

   # lat/lon lists for each model
   fhrs=[]
   lats=[]
   lons=[]
   vmax=[]
   pres=[]
   rmw=[]

 # track_file = DATA_DIR+'/atcfunix.gfs.'+cycle
   track_file = GFS_DIR+'/trak.gfso.atcfunix.altg.'+cycle

   with open(track_file,'r') as ofile:
      reader=csv.reader(ofile)
      for row in reader:
         if row[0].replace(" ","")==str.upper(TC_number[0:2]) and row[1].replace(" ","")==TC_number[2:4]+'L' and row[11].replace(" ","") == '34':
            if int(row[5]) <= 180:      # shorten tracks to F180 to match GFS a-deck
               fhrs.append(int(row[5].replace(" ","")))
               lats.append(float(re.sub("N","",row[6]))/10.0)
               try:
                  lons.append(float(re.sub("W","",row[7]))/-10.0)
               except:
                  lons.append(float(re.sub("E","",row[7]))/10.0)
               vmax.append(row[8].replace(" ",""))
               pres.append(row[9].replace(" ",""))
               rmw.append(row[19].replace(" ",""))
           

   print(len(pres))

   f = open(DATA_DIR+'/'+str.lower(TC_name)+'_'+str.lower(model_str)+'_'+cycle+'.csv','wt')

   cycle_time = datetime.datetime(int(cycle[0:4]),int(cycle[4:6]),int(cycle[6:8]),int(cycle[8:10]))

   i = 0
   try:
      writer = csv.writer(f)
      while i < len(fhrs):
         valid_time = cycle_time + datetime.timedelta(hours=int(fhrs[i]))
         writer.writerow([fhrs[i],valid_time.strftime('%Y%m%d%H'),lats[i],lons[i],pres[i],vmax[i],rmw[i]])
         i += 1


    # print cycle, cycles[i]
    # while cycle == cycles[i]:
    #    valid_time = cycle_time + datetime.timedelta(hours=int(fhrs[i]))
      #  writer.writerow([fhrs[i],valid_time.strftime('%Y%m%d%H'),lats[i],lons[i],pres[i],vmax[i],rmw[i]])
    #    writer.writerow([fhrs[i],valid_time.strftime('%Y%m%d%H'),lats[i],lons[i],pres[i],vmax[i]])
    #    i += 1

    #    if i > len(cycles)-1:
    #       break

   finally:
      f.close()





