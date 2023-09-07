#!/usr/bin/env python
# Author: L Dawson
#
# Script to pull fcst files from HPSS, rename, and save in desired data directory
# Desired cycle and model string can be passed in from command line
# If no arguments are passed in, script will prompt user for these inputs
#
# Script History Log:
# 2018-01    L Dawson  initial versioning to pull forecast files. Required command line inputs
# 2018-04-24 L Dawson  enhanced functionality by raising exceptions and adding user input prompts
# 2018-05-17 L Dawson  changed data directory declaration to be more independent

import numpy as np
import datetime, time, os, sys, subprocess


# Create data directory (if not already created)
DIR = os.getcwd() 
DIR = '/gpfs/dell2/emc/verification/noscrub/Logan.Dawson/CAM_verif/CONUSFV3/'
DIR = '/gpfs/dell2/ptmp/Logan.Dawson/MEG/FV3LAM'
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
   cycle = raw_input('Enter model initialization time (YYYYMMDDHH): ')

yyyy = int(cycle[0:4])
mm   = int(cycle[4:6])
dd   = int(cycle[6:8])
hh   = int(cycle[8:10])

date_str = datetime.datetime(yyyy,mm,dd,hh,0,0)
#cycle = date_str.strftime("%Y%m%d%H")
YYYYMMDD_CC = date_str.strftime("%Y%m%d_%H")


# Determine desired model
try:
   model_str = str(sys.argv[2])
except IndexError:
   model_str = None

if model_str is None:
   print 'Model string options: GFS, GEFS, FV3, EC, NAM, NAMNEST, RAP, HRRR, HRRRX, HIRESW, or HREF'
   model_str = raw_input('Enter desired model: ')
   ## GEFS options: GEFS, GEFSMEAN, GEFSSPREAD, GEFSCTRL, GEFSMEMS
   ## HREF options: HREF, HREFPROB, HREFMEAN, HREFPMMN, HREFAVRG
   ## HIRESW options: HIRESW, HIRESWARW, HIRESWARW2, HIRESWNMMB


# Set file prefixes/suffixes
# GFS forecasts
if str.upper(model_str) == 'GFS':
   runlength = 384

   gfs_changedate1 = datetime.datetime(2016,05,10,00,0,0)
   gfs_changedate2 = datetime.datetime(2017,07,20,00,0,0)

   if date_str < gfs_changedate1:
      TAR_PREFIX = "com_gfs_prod_gfs."
   elif (date_str >= gfs_changedate1) and (date_str <= gfs_changedate2):
      TAR_PREFIX = "com2_gfs_prod_gfs."
   elif date_str > gfs_changedate2:
      TAR_PREFIX = "gpfs_hps_nco_ops_com_gfs_prod_gfs."

   TAR_SUFFIX   = '.pgrb2_0p25.tar'
#  TAR_SUFFIX   = '.pgrb2_0p50.tar'                        # uncomment if 0.5 degree data are desired
   FILE_PREFIX  = 'gfs.t'+cycle[8:10]+'z.'
   FILE_PREFIX2 = 'gfs.'+cycle[0:8]+'.t'+cycle[8:10]+'z.'
   FILE_SUFFIX  = 'pgrb2.0p25.f'
#  FILE_SUFFIX  = 'pgrb2.0p50.f'                           # uncomment if 0.5 degree data are desired

# FV3 forecasts
elif str.upper(model_str) == 'FV3':
   runlength = 384
 # TAR_PREFIX   = 'gfs.'
   TAR_PREFIX   = 'gfsa.'
   TAR_SUFFIX   = 'tar'
   FILE_PREFIX  = 'gfs.t'+cycle[8:10]+'z.'
   FILE_PREFIX2 = 'fv3gfs.'+cycle[0:8]+'.t'+cycle[8:10]+'z.'
   FILE_SUFFIX  = 'pgrb2.0p25.f'

elif str.upper(model_str) == 'FV3TEST':
   runlength = 384
 # TAR_PREFIX   = 'gfs.'
   TAR_PREFIX   = 'gfsa.'
   TAR_SUFFIX   = 'tar'
   FILE_PREFIX  = 'gfs.t'+cycle[8:10]+'z.'
   FILE_PREFIX2 = 'fv3gfstest.'+cycle[0:8]+'.t'+cycle[8:10]+'z.'
   FILE_SUFFIX  = 'pgrb2.0p25.f'

# GEFS forecasts
elif str.upper(model_str[0:4]) == 'GEFS':
   runlength = 384
   TAR_PREFIX = 'com2_gens_prod_gefs.'

   if len(model_str) == 4:
      ens_prod = raw_input('Enter MEAN, SPREAD, CTRL, or MEMS for desired GEFS product file: ')
      model_str = model_str+ens_prod

   if str.upper(model_str[4:]) == 'MEAN':
      FILE_PREFIX  = 'geavg.t'
      FILE_PREFIX2 = 'geavg.'+cycle[0:8]+'.t'
   elif str.upper(model_str[4:]) == 'SPREAD':
      FILE_PREFIX  = 'gespr.t'
      FILE_PREFIX2 = 'gespr.'+cycle[0:8]+'.t'
   elif str.upper(model_str[4:]) == 'CTRL':
      FILE_PREFIX  = 'gec00.t'
      FILE_PREFIX2 = 'gec00.'+cycle[0:8]+'.t'

   grid_res = float(raw_input('Enter 0.5 or 1 for desired grid resolution: '))
   if grid_res == 0.5:
      RH_PREFIX = '/NCEPPROD/hpssprod/runhistory/2year/rh'+cycle[0:4]+'/'+cycle[0:6]+'/'+cycle[0:8]
      TEMP_DIR = 'pgrb2ap5'
      TAR_SUFFIX = '.pgrb2ap5.tar'
      FILE_SUFFIX = 'z.pgrb2a.0p50.f'

   elif grid_res == 1:
      RH_PREFIX = '/NCEPPROD/hpssprod/runhistory/rh'+cycle[0:4]+'/'+cycle[0:6]+'/'+cycle[0:8]
      TEMP_DIR = 'pgrb2a'
      TAR_SUFFIX = '.pgrb2a.tar'
      FILE_SUFFIX = 'z.pgrb2af'


# EC forecasts
elif str.upper(model_str) == 'EC':
   runlength = 240
   TAR_PREFIX  = 'ecm'+cycle[8:]+'_'
   TAR_SUFFIX  = cycle[0:6]+'.tar'
   FILE_PREFIX = 'pgbf'
   FILE_SUFFIX = '.ecm.'+cycle
   FILE_PREFIX2 = 'ec.'+cycle[0:8]+'.t'+cycle[8:]+'z.'
   FILE_SUFFIX2 = 'pgbf'

# NAM forecasts
elif str.upper(model_str) == 'NAM':
   runlength = 84

   nam_edate = datetime.datetime(2016,07,20,00,0,0)

   if date_str < nam_edate:
      TAR_PREFIX = 'com_nam_prod_nam.'
   elif date_str > nam_edate:
      TAR_PREFIX = 'com2_nam_prod_nam.'


   grid_res = raw_input('Enter 12, 32, phys, 3d, or native for desired grid resolution: ')
   if grid_res == '12':
      TAR_SUFFIX   = '.awip.tar'
      FILE_PREFIX  = 'nam.t'+cycle[8:10]+'z.awip12'
      FILE_PREFIX2 = 'nam.'+cycle[0:8]+'.t'+cycle[8:10]+'z.awip12'
   elif grid_res == '32':
      TAR_SUFFIX   = '.awip32.tar'
      FILE_PREFIX  = 'nam.t'+cycle[8:10]+'z.awip32'
      FILE_PREFIX2 = 'nam.'+cycle[0:8]+'.t'+cycle[8:10]+'z.awip32'
   elif str.lower(grid_res) == 'phys':
      TAR_SUFFIX   = '.awphys.tar'
      FILE_PREFIX  = 'nam.t'+cycle[8:10]+'z.awphys'
      FILE_PREFIX2 = 'nam.'+cycle[0:8]+'.t'+cycle[8:10]+'z.awphys'
   elif str.lower(grid_res) == 'native' or grid_res == 'bgrid':
      TAR_SUFFIX   = '.bgrid.tar'
      FILE_PREFIX  = 'nam.t'+cycle[8:10]+'z.awphys'
      FILE_PREFIX2 = 'nam.'+cycle[0:8]+'.t'+cycle[8:10]+'z.awphys'
   elif str.lower(grid_res) == '3d':
      TAR_SUFFIX   = '.awip.tar'
      FILE_PREFIX  = 'nam.t'+cycle[8:10]+'z.awip3d'
      FILE_PREFIX2 = 'nam.'+cycle[0:8]+'.t'+cycle[8:10]+'z.awip3d'
   else:
      raise ValueError, 'Must enter 12, 32, phys, native, or 3d for desired resolution'

   FILE_SUFFIX  = '.tm00.grib2'

# SREF forecasts
elif str.upper(model_str) == 'SREF':
   TAR_PREFIX = 'com2_sref_prod_sref.'
   TAR_SUFFIX = 'com_hiresw_prod_hiresw.'

# HREF forecasts
elif str.upper(model_str[0:4]) == 'HREF':
   runlength = 36
   TAR_PREFIX = 'gpfs_hps_nco_ops_com_hiresw_prod_href.'
   TAR_SUFFIX = '_ensprod.tar'

   if len(model_str) == 8:
      ens_prod = model_str[4:]
   elif len(model_str) == 4:
      ens_prod = raw_input('Enter PROB, MEAN, PMMN, or AVRG for desired HREF product file: ')
      model_str = model_str+ens_prod

   FILE_PREFIX  = 'href.t'+cycle[8:10]+'z.conus.'+str.lower(ens_prod)+'.f'
   FILE_PREFIX2 = 'href.'+cycle[0:8]+'.t'+cycle[8:10]+'z.conus.'+str.lower(ens_prod)+'.f'
   FILE_SUFFIX  = '.grib2'

elif str.upper(model_str) == 'NAM3' or str.upper(model_str) == 'NAMNEST':
   runlength = 60
   TAR_PREFIX   = 'com2_nam_prod_nam.'
   TAR_SUFFIX   = '.conusnest.tar'
   FILE_PREFIX  = 'nam.t'+cycle[8:10]+'z.conusnest.hiresf'
   FILE_PREFIX2 = 'nam.'+cycle[0:8]+'.t'+cycle[8:10]+'z.conusnest.hiresf'
   FILE_SUFFIX  = '.tm00.grib2'

elif str.upper(model_str[0:6]) == 'HIRESW':
   runlength = 48
   TAR_PREFIX = 'com_hiresw_prod_hiresw.'
   TAR_SUFFIX = '.5km.tar'

   if len(model_str) == 6:
      hiresw_member = raw_input('Enter ARW, ARW2, or NMMB for desired HiResW run: ')
      model_str = model_str+hiresw_member

   if str.upper(model_str[6:]) == 'NMMB':
      FILE_PREFIX  = 'hiresw.t'+cycle[8:10]+'z.nmmb_5km.f'
      FILE_PREFIX2 = 'hiresw.'+cycle[0:8]+'.t'+cycle[8:10]+'z.nmmb_5km.f'
      FILE_SUFFIX  = '.conus.grib2'
   elif str.upper(model_str[6:]) == 'ARW':
      FILE_PREFIX  = 'hiresw.t'+cycle[8:10]+'z.arw_5km.f'
      FILE_PREFIX2 = 'hiresw.'+cycle[0:8]+'.t'+cycle[8:10]+'z.arw_5km.f'
      FILE_SUFFIX  = '.conus.grib2'
   elif str.upper(model_str[6:]) == 'ARW2':
      FILE_PREFIX  = 'hiresw.t'+cycle[8:10]+'z.arw_5km.f'
      FILE_PREFIX2 = 'hiresw.'+cycle[0:8]+'.t'+cycle[8:10]+'z.arw_5km.f'
      FILE_SUFFIX  = '.conusmem2.grib2'
   elif str.upper(model_str[6:]) == 'FV3':
      TAR_PREFIX = 'fv3hrw.'
      TAR_SUFFIX = '.conus.tar'
      FILE_PREFIX  = 'hiresw.t'+cycle[8:10]+'z.fv3_5km.f'
      FILE_PREFIX2 = 'hiresw.'+cycle[0:8]+'.t'+cycle[8:10]+'z.fv3_5km.f'
      FILE_SUFFIX  = '.conus.grib2'

elif str.upper(model_str) == 'RAP' or str.upper(model_str) == 'HRRR':
 
   raphrrr_changedate = datetime.datetime(2018,07,11,00,0,0)

   if str.upper(model_str) == 'RAP':
      if date_str < raphrrr_changedate:
         TAR_PREFIX = 'com2_rap_prod_rap.'
         runlength = 21
      elif date_str > raphrrr_changedate:
         TAR_PREFIX = "gpfs_hps_nco_ops_com_rap_prod_rap."
         if cycle[8:] == '03' or cycle[8:] == '09' or cycle[8:] == '15' or cycle[8:] == '21':
            runlength = 39
         else:   
            runlength = 21

      FILE_PREFIX  = 'rap.t'+cycle[8:10]+'z.awp130pgrbf'
      FILE_PREFIX2 = 'rap.'+cycle[0:8]+'.t'+cycle[8:10]+'z.awp130pgrbf'
      TAR_SUFFIX   = '.awp130.tar'        # updated to full correct one later
      FILE_SUFFIX  = '.grib2'
   elif str.upper(model_str) == 'HRRR':
      if date_str < raphrrr_changedate:
         TAR_PREFIX = 'com2_hrrr_prod_hrrr.'
         runlength = 18
      elif date_str > raphrrr_changedate:
         TAR_PREFIX = "gpfs_hps_nco_ops_com_hrrr_prod_hrrr."
         if cycle[8:] == '00' or cycle[8:] == '06' or cycle[8:] == '12' or cycle[8:] == '18':
            runlength = 36
         else:   
            runlength = 18

      FILE_PREFIX  = 'hrrr.t'+cycle[8:10]+'z.wrfprsf'
      FILE_PREFIX2 = 'hrrr.'+cycle[0:8]+'.t'+cycle[8:10]+'z.wrfprsf'
      TAR_SUFFIX   = '.wrf.tar'        # updated to full correct one later
      FILE_SUFFIX   = '.grib2'


elif str.upper(model_str) == 'HRRRX':
   if cycle[8:] == '00' or cycle[8:] == '06' or cycle[8:] == '12' or cycle[8:] == '18':
      runlength = 36
   else:   
      runlength = 18
   TAR_PREFIX   = 'hrrr.'+cycle
   TAR_SUFFIX   = '.tar'
   FILE_PREFIX  = 'hrrr.t'+cycle[8:10]+'z.wrfprsf'
   FILE_PREFIX2 = 'hrrrx.'+cycle[0:8]+'.t'+cycle[8:10]+'z.wrfprsf'
   FILE_SUFFIX  = '.grib2'


elif str.upper(model_str) == 'FV3SAR' or str.upper(model_str) == 'FV3NEST':
   runlength = 60
   TAR_PREFIX   = str.lower(model_str)+'.'
   TAR_SUFFIX   = '.conus.tar'
   FILE_PREFIX  = str.lower(model_str)+'.t'+cycle[8:10]+'z.conus.f'
   FILE_PREFIX2 = str.lower(model_str)+'.'+cycle[0:8]+'.t'+cycle[8:10]+'z.conus.f'
   FILE_SUFFIX  = '.grib2'

elif str.upper(model_str) == 'FV3SARX':
   runlength = 60
   TAR_PREFIX   = str.lower(model_str[0:6])+'.'
   TAR_SUFFIX   = '.conus.tar'
   FILE_PREFIX  = str.lower(model_str[0:6])+'.t'+cycle[8:10]+'z.conus.f'
   FILE_PREFIX2 = str.lower(model_str)+'.'+cycle[0:8]+'.t'+cycle[8:10]+'z.conus.f'
   FILE_SUFFIX  = '.grib2'


#===========================================================================================================
# Set prefix for correct run history path
#===========================================================================================================
if str.upper(model_str) == 'GFS' or str.upper(model_str) == 'NAM' or str.upper(model_str) == 'SREF' or str.upper(model_str) == 'RAP' or str.upper(model_str) == 'HRRR':
   RH_PREFIX = '/NCEPPROD/hpssprod/runhistory/rh'+cycle[0:4]+'/'+cycle[0:6]+'/'+cycle[0:8]

elif str.upper(model_str) == 'HIRESWFV3':
   RH_PREFIX = '/NCEPDEV/emc-meso/2year/Matthew.Pyle/fv3hrw/rh'+cycle[0:4]+'/'+cycle[0:6]+'/'+cycle[0:8]

elif str.upper(model_str) == 'NAM3' or str.upper(model_str) == 'NAMNEST' or str.upper(model_str[0:4]) == 'HREF' or str.upper(model_str[0:6]) == 'HIRESW':
   RH_PREFIX = '/NCEPPROD/hpssprod/runhistory/2year/rh'+cycle[0:4]+'/'+cycle[0:6]+'/'+cycle[0:8]

elif str.upper(model_str) == 'FV3':
 # RH_PREFIX = '/NCEPDEV/emc-global/1year/emc.glopara/WCOSS_C/scratch/prfv3l65'   # old path to experiment with GFS ICs and GFDL MP
 # RH_PREFIX = '/NCEPDEV/emc-global/5year/emc.glopara/WCOSS_C/Q1FY19/prfv3rt1'    # old path to fully-cycled experiment with old implementation date
   RH_PREFIX = '/NCEPDEV/emc-global/5year/emc.glopara/WCOSS_C/Q2FY19/prfv3rt1'    # new path to fully-cycled experiment with new implementation date

elif str.upper(model_str) == 'FV3TEST':
   RH_PREFIX = '/NCEPDEV/emc-global/5year/emc.glopara/WCOSS_C/Q2FY19/prfv3rt3s'

elif str.upper(model_str) == 'EC':
  RH_PREFIX = '/NCEPDEV/emc-global/5year/emc.glopara/stats/ecm/'

elif str.upper(model_str) == 'HRRRX':
  RH_PREFIX = '/NCEPDEV/emc-meso/1year/Benjamin.Blake/hrrrv3/'

elif str.upper(model_str) == 'FV3SAR' or str.upper(model_str) == 'FV3NEST':
  RH_PREFIX = '/NCEPDEV/emc-meso/2year/Benjamin.Blake/'+str.lower(model_str)+'/rh'+cycle[0:4]+'/'+cycle[0:6]+'/'+cycle[0:8]

elif str.upper(model_str) == 'FV3SARX':
  RH_PREFIX = '/NCEPDEV/emc-meso/2year/Eric.Rogers/'+str.lower(model_str[0:6])+'/rh'+cycle[0:4]+'/'+cycle[0:6]+'/'+cycle[0:8]





#===========================================================================================================
# Retrieve list of forecast hours
#===========================================================================================================

### By default, will ask for command line input to determine which analysis files to pull 
### User can uncomment and modify the next line to bypass the command line calls
#fhrs = np.arange(0,7,6)

if cycle == '2019041012':
   fhrs = np.arange(0,25,1)
elif cycle == '2019041000':
   fhrs = np.arange(12,37,3)
elif cycle == '2019040912':
   fhrs = np.arange(24,49,3)

elif cycle == '2018062812':
   fhrs = np.arange(0,25,3)
elif cycle == '2018062806':
   fhrs = np.arange(0,31,3)
elif cycle == '2018062800':
   fhrs = np.arange(0,37,3)
elif cycle == '2018062718':
   fhrs = np.arange(0,43,3)
elif cycle == '2018062712':
   fhrs = np.arange(0,49,3)
elif cycle == '2018062706':
   fhrs = np.arange(0,55,3)
elif cycle == '2018062700':
   fhrs = np.arange(0,61,3)
elif cycle == '2018062618':
   fhrs = np.arange(42,55,6)
elif cycle == '2018062612':
   fhrs = np.arange(0,61,3)

fhrs = np.arange(0,49,1)
fhrs = [6]
#fhrs = np.arange(13,25,1)

try:
   fhrs
except NameError:
   fhrs = None

if fhrs is None:
   fhrb = int(raw_input('Enter first forecast hour (cannot be < 0): '))
   if model_str[0:4] == 'HREF' and fhrb == 0:
      fhrb = 1
      print 'No 0-h HREF forecast. First forecast hour set to 1' 

   fhre = int(raw_input('Enter last forecast hour: '))
   if fhre > runlength:
      fhre = runlength
      print 'Invalid run length. Last forecast hour set to '+str(fhre) 

   step = int(raw_input('Enter hourly step: '))
   if str.upper(model_str[0:4]) == 'GEFS' and step%6 != 0:
      raise ValueError, 'Invalid GEFS step'

   fhrs = np.arange(fhrb,fhre+1,step)


print 'Array of hours is: '
print fhrs


date_list = [date_str + datetime.timedelta(hours=x) for x in fhrs]

#===========================================================================================================

#===========================================================================================================

# Loop through
req = {'htar_ball':'','htar_fname':[],'mv_from':[],'mv_to':[]}
for nf in range(len(date_list)):

   print "Getting "+str(fhrs[nf])+"-h forecast from "+cycle+" "+model_str+" cycle"


   # GFS
   if str.upper(model_str) == 'GFS':
      htar_ball = RH_PREFIX+'/'+TAR_PREFIX+cycle+TAR_SUFFIX+' ./'+FILE_PREFIX+FILE_SUFFIX+str(fhrs[nf]).zfill(3)
      htar_fname = './'+FILE_PREFIX+FILE_SUFFIX+str(fhrs[nf]).zfill(3)+' ./'+FILE_PREFIX2+FILE_SUFFIX+str(fhrs[nf]).zfill(3) 
      mv_from = './'+FILE_PREFIX+FILE_SUFFIX+str(fhrs[nf]).zfill(3)
      mv_to = './'+FILE_PREFIX2+FILE_SUFFIX+str(fhrs[nf]).zfill(3)

#     os.system('htar -xvf '+RH_PREFIX+'/'+TAR_PREFIX+cycle+TAR_SUFFIX+' ./'+FILE_PREFIX+FILE_SUFFIX+str(fhrs[nf]).zfill(3))
#     os.system('mv ./'+FILE_PREFIX+FILE_SUFFIX+str(fhrs[nf]).zfill(3)+' ./'+FILE_PREFIX2+FILE_SUFFIX+str(fhrs[nf]).zfill(3))

   # GEFS
   elif str.upper(model_str[0:4]) == 'GEFS':
      if grid_res == 1 and fhrs[nf] < 100:
         digits = 2
      else:
         digits = 3

      if str.upper(model_str[4:7]) != 'MEM':
         os.system('htar -xvf '+RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDD_CC+TAR_SUFFIX+' ./'+TEMP_DIR+'/'+FILE_PREFIX+cycle[8:]+FILE_SUFFIX+str(fhrs[nf]).zfill(digits))
         os.system('mv ./'+TEMP_DIR+'/'+FILE_PREFIX+cycle[8:]+FILE_SUFFIX+str(fhrs[nf]).zfill(digits)+' ./'+FILE_PREFIX2+cycle[8:]+FILE_SUFFIX+str(fhrs[nf]).zfill(3))

      elif str.upper(model_str[4:7]) == 'MEM':
         for k in range(1,21):
             print "Getting "+str(fhrs[nf]).zfill(3)+"-h forecast for "+cycle+" GEFS mem "+str(k)+" ("+str(grid_res)+" deg)"
             os.system('htar -xvf '+RH_PREFIX+'/'+TAR_PREFIX+YYYYMMDD_CC+TAR_SUFFIX+ \
                       ' ./'+TEMP_DIR+'/gep'+str(k).zfill(2)+'.t'+cycle[8:]+FILE_SUFFIX+str(fhrs[nf]).zfill(digits))
             os.system('mv ./'+TEMP_DIR+'/gep'+str(k).zfill(2)+'.t'+cycle[8:]+FILE_SUFFIX+str(fhrs[nf]).zfill(digits)+ \
                       ' ./gep'+str(k).zfill(2)+'.'+cycle[0:8]+'.t'+cycle[8:]+FILE_SUFFIX+str(fhrs[nf]).zfill(3))

   # FV3
   elif str.upper(model_str) == 'FV3' or str.upper(model_str) == 'FV3TEST':
    # os.system('htar -xvf '+RH_PREFIX+'/'+cycle+'/'+TAR_PREFIX+TAR_SUFFIX+' gfs/'+FILE_PREFIX+FILE_SUFFIX+str(fhrs[nf]).zfill(3))  # old structure for GFS IC exp
    # os.system('mv gfs/'+FILE_PREFIX+FILE_SUFFIX+str(fhrs[nf]).zfill(3)+' ./'+FILE_PREFIX2+FILE_SUFFIX+str(fhrs[nf]).zfill(3))          # old structure for GFS IC exp
      os.system('htar -xvf '+RH_PREFIX+'/'+cycle+'/'+TAR_PREFIX+TAR_SUFFIX+' ./gfs.'+cycle[0:8]+'/'+cycle[8:10]+'/'+FILE_PREFIX+FILE_SUFFIX+str(fhrs[nf]).zfill(3))
      os.system('mv ./gfs.'+cycle[0:8]+'/'+cycle[8:10]+'/'+FILE_PREFIX+FILE_SUFFIX+str(fhrs[nf]).zfill(3)+' ./'+FILE_PREFIX2+FILE_SUFFIX+str(fhrs[nf]).zfill(3))

      if TAR_PREFIX == 'gfsa.':
         os.system('rm -fR ./gfs.'+cycle[0:8]+'/'+cycle[8:10]+'/')

   # NAM/NAM NEST/FV3-CAM
   elif str.upper(model_str[0:3]) == 'NAM' or str.upper(model_str[0:6]) == 'FV3SAR' or str.upper(model_str) == 'FV3NEST':
      htar_ball = RH_PREFIX+'/'+TAR_PREFIX+cycle+TAR_SUFFIX
      htar_fname = './'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX
      mv_from = './'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX
      mv_to = './'+FILE_PREFIX2+str(fhrs[nf]).zfill(2)+FILE_SUFFIX
 #    os.system('htar -xvf '+RH_PREFIX+'/'+TAR_PREFIX+cycle+TAR_SUFFIX+' ./'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX)
 #    os.system('mv ./'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX+' ./'+FILE_PREFIX2+str(fhrs[nf]).zfill(2)+FILE_SUFFIX)

   # EC
   elif str.upper(model_str) == 'EC':
      os.system('htar -xvf '+RH_PREFIX+TAR_PREFIX+TAR_SUFFIX+' '+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX)
      os.system('mv ./'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX+' ./'+FILE_PREFIX2+FILE_SUFFIX2+str(fhrs[nf]).zfill(3))

   # HRRRX
   elif str.upper(model_str) == 'HRRRX':
      os.system('htar -xvf '+RH_PREFIX+'hrrr.'+cycle[0:8]+'/'+TAR_PREFIX+TAR_SUFFIX+' ./'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX)
      os.system('mv ./'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX+' ./'+FILE_PREFIX2+str(fhrs[nf]).zfill(2)+FILE_SUFFIX)

   # RAP and HRRR
   elif str.upper(model_str) == 'RAP' or str.upper(model_str) == 'HRRR':
      if str.upper(model_str) == 'RAP':
         if int(cycle[8:]) <= 5:
            TAR_SUFFIX  = '00-05.awp130.tar'
         elif int(cycle[8:]) >= 6 and int(cycle[8:]) <= 11:
            TAR_SUFFIX  = '06-11.awp130.tar'
         elif int(cycle[8:]) >= 12 and int(cycle[8:]) <= 17:
            TAR_SUFFIX  = '12-17.awp130.tar'
         elif int(cycle[8:]) >= 18 and int(cycle[8:]) <= 23:
            TAR_SUFFIX  = '18-23.awp130.tar'

      elif str.upper(model_str) == 'HRRR':
         if int(cycle[8:]) <= 5:
            TAR_SUFFIX  = '00-05.wrf.tar'
         elif int(cycle[8:]) >= 6 and int(cycle[8:]) <= 11:
            TAR_SUFFIX  = '06-11.wrf.tar'
         elif int(cycle[8:]) >= 12 and int(cycle[8:]) <= 17:
            TAR_SUFFIX  = '12-17.wrf.tar'
         elif int(cycle[8:]) >= 18 and int(cycle[8:]) <= 23:
            TAR_SUFFIX  = '18-23.wrf.tar'

         if date_str >= raphrrr_changedate:
            TAR_SUFFIX = '_conus'+TAR_SUFFIX

      htar_ball = RH_PREFIX+'/'+TAR_PREFIX+cycle[0:8]+TAR_SUFFIX
      htar_fname = './'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX
      mv_from = './'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX
      mv_to = './'+FILE_PREFIX2+str(fhrs[nf]).zfill(2)+FILE_SUFFIX
   #  os.system('htar -xvf '+RH_PREFIX+'/'+TAR_PREFIX+cycle[0:8]+TAR_SUFFIX+' ./'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX)
   #  os.system('mv ./'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX+' ./'+FILE_PREFIX2+str(fhrs[nf]).zfill(2)+FILE_SUFFIX)


   # HIRESW FV3
   elif str.upper(model_str) == 'HIRESWFV3':
      htar_ball = RH_PREFIX+'/'+TAR_PREFIX+cycle+TAR_SUFFIX
      htar_fname = './'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX
      mv_from = './'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX
      mv_to = './'+FILE_PREFIX2+str(fhrs[nf]).zfill(2)+FILE_SUFFIX

   # HREF and HIRESW
   else:
      htar_ball = RH_PREFIX+'/'+TAR_PREFIX+cycle[0:8]+TAR_SUFFIX
      htar_fname = './'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX
      mv_from = './'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX
      mv_to = './'+FILE_PREFIX2+str(fhrs[nf]).zfill(2)+FILE_SUFFIX
   #  os.system('htar -xvf '+RH_PREFIX+'/'+TAR_PREFIX+cycle[0:8]+TAR_SUFFIX+' ./'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX)
   #  os.system('mv ./'+FILE_PREFIX+str(fhrs[nf]).zfill(2)+FILE_SUFFIX+' ./'+FILE_PREFIX2+str(fhrs[nf]).zfill(2)+FILE_SUFFIX)


   # Update dict
   req['htar_ball'] = htar_ball
   req['htar_fname'].append(htar_fname)
   req['mv_from'].append(mv_from)
   req['mv_to'].append(mv_to)


# Make temp text file to use for HPSS request
o = open("./"+str.lower(model_str)+"_"+cycle+"_temp.txt","w")
o.write('\n'.join(req['htar_fname']))
o.close()

# Submit HPSS request
os.system('htar -xvf '+req['htar_ball']+' -L '+str.lower(model_str)+'_'+cycle+'_temp.txt')

# Iterate through every item that was requested
for idx, (mv_from, mv_to) in enumerate(zip(req['mv_from'],req['mv_to'])):
    os.system("mv "+mv_from+" "+mv_to)

# Delete temp text file
os.system("rm ./"+str.lower(model_str)+"_"+cycle+"_temp.txt")

print("Done")
