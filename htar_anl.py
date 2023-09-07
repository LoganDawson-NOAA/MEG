#!/usr/bin/env python
# Author: L Dawson
#
# Script to pull analyses from HPSS, rename, and save in desired data directory
# Desired initial cycle and analysis string can be passed in from command line
# If no arguments are passed in, script will prompt user for these inputs
#
# Script History Log:
# 2018-01    L Dawson  initial versioning to pull analyses. Required command line inputs
# 2018-04-24 L Dawson  enhanced functionality by raising exceptions and adding user input prompts
# 2018-05-17 L Dawson  changed data directory declaration to be more independent

import numpy as np
import datetime, time, os, sys, subprocess


# Create data directory (if not already created)
DIR = os.getcwd() 
DIR = '/gpfs/dell2/ptmp/Logan.Dawson/florence'
DIR = '/gpfs/dell2/emc/verification/noscrub/Logan.Dawson/MEG'
DATA_DIR = os.path.join(DIR, 'data')
if not os.path.exists(DATA_DIR):
   os.makedirs(DATA_DIR)
os.chdir(DATA_DIR)


# Determine initial date/time
try:
   cycle = str(sys.argv[1])
except IndexError:
   cycle = None

if cycle is None:
   cycle = raw_input('Enter initial analysis time (YYYYMMDDHH): ')
   
yyyy = int(cycle[0:4])
mm   = int(cycle[4:6])
dd   = int(cycle[6:8])
hh   = int(cycle[8:10])

date_str = datetime.datetime(yyyy,mm,dd,hh,0,0)


# Determine desired analysis 
try:
   anl_str = str(sys.argv[2])
except IndexError:
   anl_str = None

if anl_str is None:
   print('Analysis string options: GFS, FV3, NAM, NAMNEST, RAP, HRRR, RTMA, URMA, REFD, ST4')
   anl_str = raw_input('Enter desired analysis: ')


# Set prefix for correct runhistory path
RH_PRE = '/NCEPPROD/hpssprod/runhistory/'


# Set correct tarball and file prefixes/suffixes

# Date when NCO standardized all runhistory prefixes 
nco_changedate = datetime.datetime(2020,2,28,0,0,0)

# RTMA/URMA
if str.upper(anl_str) == 'RTMA' or str.upper(anl_str) == 'URMA':

   rtmaurma_changedate = datetime.datetime(2018,12,4,0,0,0)

   if str.upper(anl_str) == 'RTMA':
      TAR_PREFIX  = 'com2_rtma_prod_rtma2p5.'
      FILE_PREFIX = 'rtma2p5.t'
   elif str.upper(anl_str) == 'URMA':
      TAR_PREFIX  = 'com2_urma_prod_urma2p5.'
      FILE_PREFIX = 'urma2p5.t'
   TAR_SUFFIX  = '.tar'        # updated to full correct one later
   FILE_SUFFIX = 'z.2dvaranl_ndfd.grb2'
   FILE_SUFFIX2 = 'z.2dvaranl_ndfd.grb2_wexp'

# Stage-IV analysis
elif str.upper(anl_str) == 'ST4':
   TAR_PREFIX  = 'com2_pcpanl_prod_pcpanl.'
   TAR_SUFFIX  = '.tar'

   st4_region = raw_input('Enter PR or AK, if desired. Else, download CONUS file: ')
   if str.upper(st4_region) == 'PR':
       FILE_PREFIX = 'st4_pr.'
   elif str.upper(st4_region) == 'AK':
       FILE_PREFIX = 'st4_ak.'
   else:    
       FILE_PREFIX = 'ST4.'


   st4_accum = raw_input('Enter 1, 6, or 24 for desired accumulation length: ')
   if int(st4_accum) == 1:
      FILE_SUFFIX   = '.01h.gz'
   elif st4_accum == '6':
      FILE_SUFFIX   = '.06h.gz'
   elif st4_accum == '24':
      FILE_SUFFIX   = '.24h.gz'
   else:
      raise ValueError('Must enter 1, 6, or 24 for desired accumulation')

# MRMS 3-D reflectivity mosaics
elif str.upper(anl_str) == 'RADAR' or str.upper(anl_str) == "REFD":
   TAR_PREFIX  = 'com_hourly_prod_radar.'
   TAR_SUFFIX  = '.save.tar'
   FILE_PREFIX = 'refd3d.t'
   FILE_SUFFIX = 'z.grb2f00'

# GFS analysis
elif str.upper(anl_str) == 'GFS':

   gfs_changedate1 = datetime.datetime(2016,5,10,0,0,0)
   gfs_changedate2 = datetime.datetime(2017,7,20,0,0,0)

   if date_str < gfs_changedate1 or date_str >= nco_changedate:
      TAR_PREFIX = "com_gfs_prod_gfs."
   elif (date_str >= gfs_changedate1) and (date_str <= gfs_changedate2):
      TAR_PREFIX = "com2_gfs_prod_gfs."
   elif date_str > gfs_changedate2:
      TAR_PREFIX = "gpfs_hps_nco_ops_com_gfs_prod_gfs."


   if date_str >= nco_changedate:
       TAR_SUFFIX   = '.gfs_pgrb2.tar'
   else:
       TAR_SUFFIX   = '.pgrb2_0p25.tar'

   ## File prefixes correctly set later

# FV3 analysis
elif str.upper(anl_str) == 'FV3':
   TAR_FILE = 'gfs.tar'

   ## File prefixes correctly set later

# EC analysis
elif str.upper(anl_str) == 'EC':
   TAR_PREFIX  = 'ecm'
   TAR_SUFFIX  = '.tar'

   ## File prefixes correctly set later

# NAM analysis
elif str.upper(anl_str) == 'NAM':
   TAR_PREFIX   = 'com2_nam_prod_nam.'

   nam_resolution = raw_input('Enter 12 or 32 for desired file resolution: ')
   if nam_resolution == '12':
      TAR_SUFFIX   = '.awip.tar'
   elif nam_resolution == '32':
      TAR_SUFFIX   = '.awip32.tar'
   else:
      raise ValueError('Must enter 12 or 32 for desired resolution')

   ##File prefixes correctly set later

# NAM Nest analysis
elif str.upper(anl_str) == 'NAM3' or str.upper(anl_str) == 'NAMNEST':

   if date_str >= nco_changedate:
      TAR_PREFIX = 'com_nam_prod_nam.'
   elif date_str < nco_changedate:
      TAR_PREFIX = 'com2_nam_prod_nam.'

   TAR_SUFFIX   = '.conusnest.tar'

# RAP/HRRR analyses
elif str.upper(anl_str) == 'RAP' or str.upper(anl_str) == 'HRRR':

   raphrrr_changedate = datetime.datetime(2018,7,11,0,0,0)

   if str.upper(anl_str) == 'RAP':
      if date_str < raphrrr_changedate:
         TAR_PREFIX = 'com2_rap_prod_rap.'
      elif date_str > raphrrr_changedate:
         TAR_PREFIX = "gpfs_hps_nco_ops_com_rap_prod_rap."

      FILE_PREFIX = 'rap.t'
      TAR_SUFFIX  = '.awp130.tar'        # updated to full correct one later
      FILE_SUFFIX = 'z.awp130pgrbf00.grib2'
   elif str.upper(anl_str) == 'HRRR':
      if date_str < raphrrr_changedate:
         TAR_PREFIX = 'com2_hrrr_prod_hrrr.'
      elif date_str > raphrrr_changedate:
         TAR_PREFIX = "gpfs_hps_nco_ops_com_hrrr_prod_hrrr."

      FILE_PREFIX = 'hrrr.t'
      TAR_SUFFIX  = '.wrf.tar'        # updated to full correct one later
      FILE_SUFFIX = 'z.wrfprsf00.grib2'


#===========================================================================================================
# Retrieve list of analysis hours
#===========================================================================================================

# By default, will ask for command line input to determine which analysis files to pull 
# User can uncomment and modify the next line to bypass the command line calls
#nhrs = np.arange(0,7,6)

try:
   nhrs
except NameError:
   nhrs = None

if nhrs is None:
   hrb = int(input('Enter first hour (normally 0): '))
   hre = int(input('Enter last hour: '))
   step = int(input('Enter hourly step: '))
   nhrs = np.arange(hrb,hre+1,step)


print('Array of hours is: ')
print(nhrs)

date_list = [date_str + datetime.timedelta(hours=int(x)) for x in nhrs]


#===========================================================================================================

#===========================================================================================================

# Loop through date list to extract and rename analysis files
req = {'htar_ball':'','htar_fname':[],'mv_from':[],'mv_to':[]}
for j in range(len(date_list)):

   YYYYMMDDCC = date_list[j].strftime("%Y%m%d%H")
   YYYYMMDD_CC = date_list[j].strftime("%Y%m%d_%H")
   print("Getting "+YYYYMMDDCC+" "+anl_str+" analysis")

   if str.upper(anl_str) != 'FV3' and str.upper(anl_str) != 'NAM3' and str.upper(anl_str) != 'EC':
      RH_PREFIX = RH_PRE+'rh'+YYYYMMDDCC[0:4]+'/'+YYYYMMDDCC[0:6]+'/'+YYYYMMDDCC[0:8]

   if str.upper(anl_str) == 'ST4':
      htar_ball = RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDDCC[0:8]+TAR_SUFFIX
      htar_fname = './'+FILE_PREFIX+YYYYMMDDCC+FILE_SUFFIX
      mv_from = './'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX
      mv_to = './'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX

    # os.system('htar -xvf '+RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDDCC[0:8]+TAR_SUFFIX+' ./'+FILE_PREFIX+YYYYMMDDCC+FILE_SUFFIX)
    # os.system('gunzip ./'+FILE_PREFIX+YYYYMMDDCC+FILE_SUFFIX)

   elif str.upper(anl_str) == 'RADAR' or str.upper(anl_str) == 'REFD':
      FILE_PREFIX2 = 'refd3d.'+YYYYMMDDCC[0:8]+'.t'
      htar_ball = RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDDCC[0:8]+TAR_SUFFIX
      htar_fname = './'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX
      mv_from = './'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX
      mv_to = './'+FILE_PREFIX2+YYYYMMDDCC[8:]+FILE_SUFFIX

   #  os.system('htar -xvf '+RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDDCC[0:8]+TAR_SUFFIX+' ./'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX)
   #  os.system('mv ./'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX+' ./'+FILE_PREFIX2+YYYYMMDDCC[8:]+FILE_SUFFIX)

   elif str.upper(anl_str) == 'RTMA' or str.upper(anl_str) == 'URMA':
      if str.upper(anl_str) == 'RTMA':
         FILE_PREFIX2 = 'rtma2p5.'+YYYYMMDDCC[0:8]+'.t'
      elif str.upper(anl_str) == 'URMA':
         FILE_PREFIX2 = 'urma2p5.'+YYYYMMDDCC[0:8]+'.t'

      if int(YYYYMMDDCC[8:]) <= 5:
         TAR_SUFFIX  = '00-05.tar'
      elif int(YYYYMMDDCC[8:]) >= 6 and int(YYYYMMDDCC[8:]) <= 11:
         TAR_SUFFIX  = '06-11.tar'
      elif int(YYYYMMDDCC[8:]) >= 12 and int(YYYYMMDDCC[8:]) <= 17:
         TAR_SUFFIX  = '12-17.tar'
      elif int(YYYYMMDDCC[8:]) >= 18 and int(YYYYMMDDCC[8:]) <= 23:
         TAR_SUFFIX  = '18-23.tar'

      if date_str < rtmaurma_changedate:
          htar_ball = RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDDCC[0:8]+TAR_SUFFIX
          htar_fname = './'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX
          mv_from = './'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX
          mv_to = './'+FILE_PREFIX2+YYYYMMDDCC[8:]+FILE_SUFFIX

          os.system('htar -xvf '+RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDDCC[0:8]+TAR_SUFFIX+' ./'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX)
          os.system('mv ./'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX+' ./'+FILE_PREFIX2+YYYYMMDDCC[8:]+FILE_SUFFIX)
      elif date_str > rtmaurma_changedate:
          htar_ball = RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDDCC[0:8]+TAR_SUFFIX
          htar_fname = './'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX2
          mv_from = './'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX2
          mv_to = './'+FILE_PREFIX2+YYYYMMDDCC[8:]+FILE_SUFFIX

          os.system('htar -xvf '+RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDDCC[0:8]+TAR_SUFFIX+' ./'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX2)
          os.system('mv ./'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX2+' ./'+FILE_PREFIX2+YYYYMMDDCC[8:]+FILE_SUFFIX)
      


   elif str.upper(anl_str) == 'GFS':
      mv_to = 'gfs.'+YYYYMMDDCC[0:8]+'.t'+YYYYMMDDCC[8:10]+'z.pgrb2.0p25.f000'
      if date_str >= nco_changedate:
          htar_ball = RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDD_CC+TAR_SUFFIX
          htar_fname  = 'gfs.'+YYYYMMDDCC[0:8]+'/'+YYYYMMDDCC[8:10]+'/atmos/gfs.t'+YYYYMMDDCC[8:10]+'z.pgrb2.0p25.f000'
          mv_from  = 'gfs.'+YYYYMMDDCC[0:8]+'/'+YYYYMMDDCC[8:10]+'/atmos/gfs.t'+YYYYMMDDCC[8:10]+'z.pgrb2.0p25.f000'
      else:
          htar_ball = RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDDCC+TAR_SUFFIX
          htar_fname  = 'gfs.t'+YYYYMMDDCC[8:10]+'z.pgrb2.0p25.f000'
          mv_from  = 'gfs.t'+YYYYMMDDCC[8:10]+'z.pgrb2.0p25.f000'
      os.system('htar -xvf '+htar_ball+' ./'+htar_fname)
      os.system('mv ./'+mv_from+' ./'+mv_to)

   elif str.upper(anl_str) == 'EC':
      RH_PREFIX = '/NCEPDEV/emc-global/5year/emc.glopara/stats/ecm/'
 # TAR_PREFIX  = 'ecm'
 # TAR_SUFFIX  = '.tar'
      FILE  = 'pgbf00.ecm.'+YYYYMMDDCC
      FILE2 = 'ec.'+YYYYMMDDCC[0:8]+'.t'+YYYYMMDDCC[8:10]+'z.pgbf000'
      os.system('htar -xvf '+RH_PREFIX+TAR_PREFIX+YYYYMMDDCC[8:10]+'_'+YYYYMMDDCC[0:6]+TAR_SUFFIX+' '+FILE)
      os.system('mv ./'+FILE+' ./'+FILE2)


   elif str.upper(anl_str) == 'FV3':
    # RH_PREFIX = '/NCEPDEV/emc-global/1year/emc.glopara/WCOSS_C/scratch/prfv3l65'   # old path to experiment with GFS ICs and GFDL MP
    # RH_PREFIX = '/NCEPDEV/emc-global/5year/emc.glopara/WCOSS_C/Q1FY19/prfv3rt1'    # new path to fully-cycled experiment
      RH_PREFIX = '/NCEPDEV/emc-global/5year/emc.glopara/WCOSS_C/Q2FY19/prfv3rt1'    # new path to fully-cycled experiment
      FILE  = 'gfs.t'+YYYYMMDDCC[8:10]+'z.pgrb2.0p25.f000'
      FILE2 = 'fv3.'+YYYYMMDDCC[0:8]+'.t'+YYYYMMDDCC[8:10]+'z.pgrb2.0p25.f000'
    # os.system('htar -xvf '+RH_PREFIX+'/'+YYYYMMDDCC+'/'+TAR_FILE+' gfs/'+FILE)
    # os.system('mv gfs/'+FILE+' ./'+FILE2)
      os.system('htar -xvf '+RH_PREFIX+'/'+YYYYMMDDCC+'/'+TAR_FILE+ \
                ' ./gfs.'+YYYYMMDDCC[0:8]+'/'+YYYYMMDDCC[8:10]+'/'+FILE)
      os.system('mv ./gfs.'+cycle[0:8]+'/'+cycle[8:10]+'/'+FILE+ \
                ' ./'+FILE2)

      if TAR_PREFIX == 'gfsa.':
         os.system('rm -fR ./gfs.'+cycle[0:8]+'/'+cycle[8:10]+'/')

   elif str.upper(anl_str) == 'NAM':
      if nam_resolution == '12':
         FILE  = 'nam.t'+YYYYMMDDCC[8:10]+'z.awip1200.tm00.grib2'
         FILE2 = 'nam.'+YYYYMMDDCC[0:8]+'.t'+YYYYMMDDCC[8:10]+'z.awip1200.tm00.grib2'
      elif nam_resolution == '32':
         FILE  = 'nam.t'+YYYYMMDDCC[8:10]+'z.awip3200.tm00.grib2'
         FILE2 = 'nam.'+YYYYMMDDCC[0:8]+'.t'+YYYYMMDDCC[8:10]+'z.awip3200.tm00.grib2'
      os.system('htar -xvf '+RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDDCC+TAR_SUFFIX+' ./'+FILE)
      os.system('mv ./'+FILE+' ./'+FILE2)

   elif str.upper(anl_str) == 'NAM3':
      RH_PREFIX = RH_PRE+'2year/rh'+YYYYMMDDCC[0:4]+'/'+YYYYMMDDCC[0:6]+'/'+YYYYMMDDCC[0:8]
      FILE  = 'nam.t'+YYYYMMDDCC[8:10]+'z.conusnest.hiresf00.tm00.grib2'
      FILE2 = 'nam.'+YYYYMMDDCC[0:8]+'.t'+YYYYMMDDCC[8:10]+'z.conusnest.hiresf00.tm00.grib2'
      os.system('htar -xvf '+RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDDCC+TAR_SUFFIX+' ./'+FILE)
      os.system('mv ./'+FILE+' ./'+FILE2)


   elif str.upper(anl_str) == 'RAP' or str.upper(anl_str) == 'HRRR':
      if str.upper(anl_str) == 'RAP':
         FILE_PREFIX2 = 'rap.'+YYYYMMDDCC[0:8]+'.t'
         if date_str > raphrrr_changedate:
            RH_PREFIX = RH_PRE+'2year/rh'+YYYYMMDDCC[0:4]+'/'+YYYYMMDDCC[0:6]+'/'+YYYYMMDDCC[0:8]

         if int(YYYYMMDDCC[8:]) <= 5:
            TAR_SUFFIX  = '00-05.awp130.tar'
         elif int(YYYYMMDDCC[8:]) >= 6 and int(YYYYMMDDCC[8:]) <= 11:
            TAR_SUFFIX  = '06-11.awp130.tar'
         elif int(YYYYMMDDCC[8:]) >= 12 and int(YYYYMMDDCC[8:]) <= 17:
            TAR_SUFFIX  = '12-17.awp130.tar'
         elif int(YYYYMMDDCC[8:]) >= 18 and int(YYYYMMDDCC[8:]) <= 23:
            TAR_SUFFIX  = '18-23.awp130.tar'

      elif str.upper(anl_str) == 'HRRR':
         FILE_PREFIX2 = 'hrrr.'+YYYYMMDDCC[0:8]+'.t'

         if int(YYYYMMDDCC[8:]) <= 5:
            TAR_SUFFIX  = '00-05.wrf.tar'
         elif int(YYYYMMDDCC[8:]) >= 6 and int(YYYYMMDDCC[8:]) <= 11:
            TAR_SUFFIX  = '06-11.wrf.tar'
         elif int(YYYYMMDDCC[8:]) >= 12 and int(YYYYMMDDCC[8:]) <= 17:
            TAR_SUFFIX  = '12-17.wrf.tar'
         elif int(YYYYMMDDCC[8:]) >= 18 and int(YYYYMMDDCC[8:]) <= 23:
            TAR_SUFFIX  = '18-23.wrf.tar'
      
      htar_ball = RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDDCC[0:8]+TAR_SUFFIX
      htar_fname = './'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX
      mv_from = './'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX
      mv_to = './'+FILE_PREFIX2+YYYYMMDDCC[8:]+FILE_SUFFIX

      os.system('htar -xvf '+RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDDCC[0:8]+TAR_SUFFIX+' ./'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX)
      os.system('mv ./'+FILE_PREFIX+YYYYMMDDCC[8:]+FILE_SUFFIX+' ./'+FILE_PREFIX2+YYYYMMDDCC[8:]+FILE_SUFFIX)


   # Update dict
   req['htar_ball'] = htar_ball
   req['htar_fname'].append(htar_fname)
   req['mv_from'].append(mv_from)
   req['mv_to'].append(mv_to)


   # Make temp text file to use for HPSS request
   o = open("./"+str.lower(anl_str)+"_"+YYYYMMDDCC+"_temp.txt","w")
   o.write('\n'.join(req['htar_fname']))
   o.close()

   # Submit HPSS request
   if str.upper(anl_str) != 'RAP' and str.upper(anl_str) != 'HRRR' and str.upper(anl_str) != 'RTMA' and str.upper(anl_str) != 'URMA': 
       os.system('htar -xvf '+req['htar_ball']+' -L '+str.lower(anl_str)+'_'+YYYYMMDDCC+'_temp.txt')

   # Iterate through every item that was requested
   if anl_str != 'ST4':
        for idx, (mv_from, mv_to) in enumerate(zip(req['mv_from'],req['mv_to'])):
            os.system("mv "+mv_from+" "+mv_to)
   else:
        for zipfile in req['htar_fname']:
            if os.path.exists(zipfile):
                os.system('gunzip '+zipfile)
                print('gunzip '+zipfile)

   # Delete temp text file
   os.system("rm ./"+str.lower(anl_str)+"_"+YYYYMMDDCC+"_temp.txt")

print("Done")
