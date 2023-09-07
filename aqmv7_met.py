#!/usr/bin/env python

import os, sys, datetime, time, subprocess
import multiprocessing, itertools, collections
import numpy as np
import grib2io
import scipy, ncepy
from ncepgrib2 import Grib2Encode, Grib2Decode
import netCDF4
import dawsonpy

import pyproj
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cartopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as image
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib import colors as c




def cmap_t2m():
 # Create colormap for 2-m temperature
 # Modified version of the ncl_t2m colormap from Jacob's ncepy code
    r=np.array([255,128,0,  70, 51, 0,  255,0, 0,  51, 255,255,255,255,255,171,128,128,36,162,255])
    g=np.array([0,  0,  0,  70, 102,162,255,92,128,185,255,214,153,102,0,  0,  0,  68, 36,162,255])
    b=np.array([255,128,128,255,255,255,255,0, 0,  102,0,  112,0,  0,  0,  56, 0,  68, 36,162,255])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=float(i)/(float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_t2m_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_T2M_COLTBL',colorDict)
    return cmap_t2m_coltbl


def cmap_svr():
 # Create colormap for 2-m temperature
    r=np.array([])
    g=np.array([])
    b=np.array([])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_svr_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_SVR_COLTBL',colorDict)
    return cmap_svr_coltbl



# Set path and create graphx directory (if not already created)
case = 'aqmv7'


# Get machine
#machine, hostname = dawsonpy.get_machine()


# Set up working directory
DATA_DIR = os.path.join('/lfs/h2/emc/vpppg/noscrub/logan.dawson','MEG','data')
GRAPHX_DIR = os.path.join('/lfs/h2/emc/ptmp/'+os.environ['USER'],'MEG', case, 'graphx')


if os.path.exists(DATA_DIR):
   if not os.path.exists(GRAPHX_DIR):
      os.makedirs(GRAPHX_DIR)
else:
 # NameError, 'data for '+case+' case not found'
   print('data for '+case+' case may not be found')
#  DIR = os.getcwd()
#  GRAPHX_DIR = os.path.join(DIR,'graphx')


# Define where Cartopy maps are located
cartopy.config['data_dir'] = '/apps/ops/prod/data/cartopy'


#Determine initial date/time
try:
   cycle = str(sys.argv[1])
except IndexError:
   cycle = None

if cycle is None:
   cycle = input('Enter initial time (YYYYMMDDHH): ')

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
   model_str = input('Enter desired model: ')
   ## GEFS options: GEFS, GEFSMEAN, GEFSSPREAD, GEFSCTRL, GEFSMEMS
   ## HREF options: HREF, HREFPROB, HREFMEAN, HREFPMMN, HREFAVRG
   ## HIRESW options: HIRESW, HIRESWARW, HIRESWARW2, HIRESWNMMB


if str.upper(model_str[0:4]) == 'HREF':
    OUT_DIR = os.path.join(GRAPHX_DIR, 'HREF', cycle)
else:
    OUT_DIR = os.path.join(GRAPHX_DIR, model_str, cycle)

if not os.path.exists(OUT_DIR):
      os.makedirs(OUT_DIR)



HiResModels = ['NAM3','HIRE','HRRR','HREF','FV3S','FV3N','FV3L']
FV3CAMs = ['FV3SAR','FV3SARX','FV3NEST','FV3SARDA','FV3LAM','FV3LAMX','FV3LAMDA','FV3LAMDAX']
ARWs = ['HIRESWARW','HIRESWARW2','HRRR','RAP']
NMMBs = ['NAM3','HIRESWNMMB']

large_domains = ['conus','Australia']
regional_domains = ['eCONUS','eCONUSxE','scCONUS','MIDATLxNE','westcoast']
subregional_domains = ['SPL','MIDSOUTH','Barry',]
state_domains = ['OK','Louisiana','dc-nyc','box-nyc','WA-BC','Seattle','ct-deep']

### By default, will ask for command line input to determine which analysis files to pull 
### User can uncomment and modify the next line to bypass the command line calls
#if str.upper(model_str) == 'NAM3':
#   fhrs = np.arange(1,61,1)
#elif str.upper(model_str[0:6]) == 'HIRESW':
#   fhrs = np.arange(1,49,1)
#elif str.upper(model_str[0:4]) == 'HREF':
#   fhrs = np.arange(1,37,1)
#elif str.upper(model_str) == 'HRRR':
#   if HH == 00 or HH == 06 or HH == 12 or HH == 18:
#      fhrs = np.arange(1,37,1)
#   else:
#      fhrs = np.arange(1,19,1)

#fhrs=[12,24]


if model_str == 'EC':
    step = 6
else:
    step = 3

if cycle == '2019051712':
   fhrs = [84]
elif cycle == '2019051812':
   fhrs = [60]
elif cycle == '2019051912':
   fhrs = np.arange(24,49,3)
#  fhrs = [36]


elif cycle == '2021102100':
   fhrs = np.arange(0,85,step)
elif cycle == '2021102112':
   fhrs = np.arange(0,73,step)
elif cycle == '2021102200':
   fhrs = np.arange(0,61,step)
elif cycle == '2021102212':
   fhrs = np.arange(0,49,step)
elif cycle == '2021102300':
   fhrs = np.arange(0,37,step)
elif cycle == '2021102312':
   fhrs = np.arange(0,25,step)
elif cycle == '2021102400':
   fhrs = np.arange(0,13,step)


elif cycle == '2021102400':
   fhrs = np.arange(0,85,step)
elif cycle == '2021102412':
   fhrs = np.arange(0,73,step)
elif cycle == '2021102500':
   fhrs = np.arange(0,61,step)
elif cycle == '2021102512':
   fhrs = np.arange(0,49,step)
elif cycle == '2021102600':
   fhrs = np.arange(0,37,step)
elif cycle == '2021102612':
   fhrs = np.arange(0,25,step)
elif cycle == '2021102700':
   fhrs = np.arange(0,13,step)

elif cycle == '2023041406':
   fhrs = np.arange(1,25,1)
elif cycle == '2023041412':
   fhrs = np.arange(1,19,1)

#fhrs = np.arange(12,13,1)
#fhrs = np.arange(0,25,1)
#fhrs = [1]

try:
   fhrs
except NameError:
   fhrs = None

if fhrs is None:
   fhrb = int(input('Enter first forecast hour (cannot be < 0): '))
#  if model_str[0:4] == 'HREF' and fhrb == 0:
#     fhrb = 1
#     print('No 0-h HREF forecast. First forecast hour set to 1')

   fhre = int(input('Enter last forecast hour: '))
#  if fhre > runlength:
#     fhre = runlength
#     print('Invalid run length. Last forecast hour set to '+str(fhre))

   step = int(input('Enter hourly step: '))
   if str.upper(model_str[0:4]) == 'GEFS' and step%6 != 0:
      raise ValueError('Invalid GEFS step')

   fhrs = np.arange(fhrb,fhre+1,step)


if str.upper(model_str[0:4]) == 'HREF' and fhrs[0] == 0:
    fhrs = np.delete(fhrs,0)


print('Array of hours is: ')
print(fhrs)


accum_length = len(fhrs)-1   # works for hourly accumulation of UH
accum_length = 24



date_list = [date_str + datetime.timedelta(hours=int(x)) for x in fhrs]


#Point locations
plot_loc = False
olats=[ 37.65,  39.13]
olons=[-97.43, -96.67]
olats=[ 39.07]
olons=[-95.62]

#Specify plots to make
#domains = ['MIDATL','Florence']
domains = ['MIDSOUTHzoom']
domains = ['DC-NYC','CONUS']
domains = ['CONUS','MIDATLxNE']
domains = ['conus','box-nyc','ct-deep']

if case == 'eastcoast':
    domains = ['CONUS','eCONUSxE','BOX-NYC']
elif case == 'westcoast':
    domains = ['CONUS','westcoast','WA-BC','Seattle']

#fields = ['mlcape_mlcin_shear06']
#fields = ['mlcape_mlcin','refc','slp_mucape_shear06']
#fields = ['omega','dzdt','rh']
#fields = ['rh800','rh825','rh850','rh875','rh900','omega800','omega825','omega850','omega875','omega900','apcp','acpcp','refc','refd1','refd4']
#fields = ['rh850','refc','refd1','refd4']
#fields = ['tcolcw','tcolci','tcolr','tcols','tcolc','tcolw','tcoli']
#fields = ['slp_mucape','sbcape_sbcin','refc_uh','td2_10mwind']
#fields = ['wind500','vort500']
#fields = ['refc_uh']
#fields = ['refc_uh','lr75','sbcape_sbcin','mucape_shear06','td2_10mwind','t2_10mwind']
#fields = ['prob_refc40']
fields = ['refc','refc_uh','apcp','apcp24','sbcape_sbcin','slp_mucape_shear06','td2_10mwind','t2_10mwind']
fields = ['refc','refc_uh','sbcape_sbcin','slp_mucape_shear06','td2_10mwind','t2_10mwind']
#fields = ['apcp','apcp24']
fields = ['refc','uh25','vv01']
fields = ['t2_10mwind']
fields = ['prob_refc40']
fields = ['10mwind']
fields = ['t2','td2']

fields = ['prob_ffri']
fields = ['wind500','vort500','500hght_slp_10mwind','10mwind','t2','td2']
fields = ['refc','sbcape_sbcin','slp_mucape_shear06','td2_10mwind','t2_10mwind']
fields = ['t2_10mwind']

accums = []
accums = ['apcp24']
accums = ['UH25_accum','UH03_accum','VV02_accum','VV01_accum']
accums = []
UH25_accum = []
UH03_accum = []
UH02_accum = []
VV02_accum = []
VV01_accum = []
APCP = []

#for nfields,ndomains in itertools.product(fields,domains):

#nplots = [zip(x,fields) for x in itertools.permutations(domains,len(fields))]
#plots = list(itertools.chain(*nplots))
#plots = [y for x in nplots for y in x]
#print(len(plots))




def main():

    plots = [n for n in itertools.product(domains,fields)]
    print(plots)

    pool = multiprocessing.Pool(len(plots))
    pool.map(plot_fields,plots)



def plot_fields(plot):

   thing = np.asarray(plot)
   domain = thing[0]
   field = thing[1]
#  print thing
#  print domain, field 

   print('plotting '+str.upper(model_str)+' '+str.upper(field)+' on '+domain+' domain')


   fig = plt.figure(figsize=(10.9,8.9))
   gs = GridSpec(1,1,wspace=0.0,hspace=0.0)


   back_res='50m'
   back_img='off'


   time1 = time.perf_counter()

   if domain == 'conus':
       print('specify new corner vals for CONUS')
       llcrnrlon=-121.5
       llcrnrlat=22.0
       urcrnrlon=-64.5
       urcrnrlat=48.0
       lon0 = -101.
   elif domain == 'nw':
       llcrnrlat=36.
       llcrnrlon=-126.
       urcrnrlat=53.
       urcrnrlon=-108.
   elif domain == 'nc':
       llcrnrlat=36.
       llcrnrlon=-112.
       urcrnrlat=52.5
       urcrnrlon=-82.
   elif domain == 'ne':
       llcrnrlat=39.
       llcrnrlon=-89.
       urcrnrlat=49.
       urcrnrlon=-63.
   elif domain == 'sw':
       llcrnrlat=22.
       llcrnrlon=-122.
       urcrnrlat=41.
       urcrnrlon=-106.
   elif domain == 'sc':
       llcrnrlat=24.
       llcrnrlon=-109.
       urcrnrlat=41.
       urcrnrlon=-85.
     # llcrnrlon = -109.0
     # llcrnrlat = 25.0
     # urcrnrlon = -88.5
     # urcrnrlat = 37.5
     # cen_lat = 31.25
     # cen_lon = -101.0
     # xextent=-529631
     # yextent=-407090
     # offset=0.25

   elif domain == 'se':
       llcrnrlat=24.
       llcrnrlon=-91.
       urcrnrlat=38.
       urcrnrlon=-68.
   elif domain == 'midatl':
       llcrnrlat=34.0
       llcrnrlon=-85.5
       urcrnrlat=41.0
       urcrnrlon=-71.25
   elif domain == 'glakes':
       llcrnrlat=40.5
       llcrnrlon=-93.5
       urcrnrlat=48.5
       urcrnrlon=-74.2
   elif domain == 'ca':
       llcrnrlat=32.
       llcrnrlon=-124.
       urcrnrlat=42.
       urcrnrlon=-117.


   elif domain == 'dc-nyc':
       llcrnrlon=-80.
       llcrnrlat=36.0
       urcrnrlon=-71.
       urcrnrlat=42.
       lon0 = -75.5
   elif domain == 'box-nyc':
       llcrnrlon=-77.6
       llcrnrlat=38.0
       urcrnrlon=-67.5
       urcrnrlat=45.
       lon0 = -73.
   elif domain == 'ct-deep':
       llcrnrlon=-75.
       llcrnrlat=40.4
       urcrnrlon=-71.
       urcrnrlat=42.2
       lon0 = -73.


   lat0 = 39.

   lat0 = (llcrnrlat + urcrnrlat)*0.5
   lon0 = (llcrnrlon + urcrnrlon)*0.5

   standard_parallels = (32,46)

   extent = [llcrnrlon-1,urcrnrlon+1,llcrnrlat-1,urcrnrlat+1]

   myproj = ccrs.LambertConformal(central_longitude=lon0,central_latitude=lat0,
            false_easting=0.0,false_northing=0.0,secant_latitudes=None,
            standard_parallels=standard_parallels,globe=None)

#  extent = [llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat]
#  myproj=ccrs.LambertConformal(central_longitude=cen_lon, central_latitude=cen_lat, false_easting=0.0,
#                               false_northing=0.0, secant_latitudes=None, standard_parallels=None,
#                               globe=None)


   # All lat lons are earth relative, so setup the associated projection correct for that data
   transform = ccrs.PlateCarree()

   ax = fig.add_subplot(gs[0:1,0:1], projection=myproj)
   ax.set_extent(extent,crs=transform)
 # ax.set_extent(extent)
   axes = [ax]

   fline_wd = 1.0        # line width
   fline_wd_lakes = 0.3  # line width
   falpha = 0.7          # transparency


   lakes = cfeature.NaturalEarthFeature('physical','lakes',back_res,
                edgecolor='black',facecolor='none',
                linewidth=fline_wd_lakes,zorder=1)
   coastlines = cfeature.NaturalEarthFeature('physical','coastline',
                back_res,edgecolor='black',facecolor='none',
                linewidth=fline_wd,zorder=1)
   states = cfeature.NaturalEarthFeature('cultural','admin_1_states_provinces',
                back_res,edgecolor='black',facecolor='none',
                linewidth=fline_wd,zorder=1)
   borders = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                back_res,edgecolor='black',facecolor='none',
                linewidth=fline_wd,zorder=1)


   # high-resolution background images
   if back_img=='on':
       img = plt.imread('/lfs/h2/emc/vpppg/save/'+os.environ['USER']+'/python/NaturalEarth/raster_files/NE1_50M_SR_W.tif')
       ax.imshow(img, origin='upper', transform=transform)


   ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidths=0.6,linestyle='solid',edgecolor='k',zorder=4)

   ax.add_feature(cfeature.OCEAN.with_scale('50m'),linewidths=0.5,linestyle='solid',edgecolor='k',facecolor='none',zorder=1)
   ax.add_feature(cfeature.LAND.with_scale('50m'),linewidths=0.5,linestyle='solid',edgecolor='k',facecolor='none',zorder=1)
   ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidths=0.5,linestyle='solid',edgecolor='k',zorder=1)
   ax.add_feature(cfeature.STATES.with_scale('50m'),linewidths=0.3,linestyle='solid',edgecolor='k',zorder=1)





   '''
   if domain == 'CONUS':
      m = Basemap(llcrnrlon=-121.5,llcrnrlat=22.,urcrnrlon=-64.5,urcrnrlat=48.,\
                  resolution='i',projection='lcc',\
                  lat_1=32.,lat_2=46.,lon_0=-101.,area_thresh=1000.,ax=ax)

   elif domain == 'eCONUS':
      m = Basemap(llcrnrlon=-105.,llcrnrlat=22.,urcrnrlon=-64.5,urcrnrlat=48.,\
                  resolution='i',projection='lcc',\
                  lat_1=32.,lat_2=46.,lon_0=-95.,area_thresh=1000.,ax=ax)

   elif domain == 'eCONUSxE':
      m = Basemap(llcrnrlon=-100.,llcrnrlat=24.,urcrnrlon=-55,urcrnrlat=50.,\
                  resolution='i',projection='lcc',\
                  lat_1=32.,lat_2=46.,lon_0=-87.,area_thresh=1000.,ax=ax)

   elif domain == 'westcoast':
      m = Basemap(llcrnrlon=-155.,llcrnrlat=36.,urcrnrlon=-100.,urcrnrlat=62.,\
                  resolution='i',projection='lcc',\
                  lat_1=32.,lat_2=46.,lon_0=-137.5,area_thresh=1000.,ax=ax)

   elif domain == 'CONUSxE':
      m = Basemap(llcrnrlon=-121.5,llcrnrlat=22.,urcrnrlon=-50,urcrnrlat=55.,\
                  resolution='i',projection='lcc',\
                  lat_1=30.,lat_2=48.,lon_0=-95.,area_thresh=1000.,ax=ax)

   elif domain == 'SE':
      m = Basemap(llcrnrlon=-95.,llcrnrlat=24.5,urcrnrlon=-75.,urcrnrlat=40.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-87.,area_thresh=1000.,ax=ax)

   elif str.upper(domain) == 'MICHAEL':
      draw_counties=False
      m = Basemap(llcrnrlon=-90.,llcrnrlat=17.5,urcrnrlon=-60.,urcrnrlat=45.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-85.,area_thresh=1000.,ax=ax)

   elif str.upper(domain) == 'BARRY':
      draw_counties=False
    # m = Basemap(llcrnrlon=-100.,llcrnrlat=26.5,urcrnrlon=-82.5,urcrnrlat=37.,\
      m = Basemap(llcrnrlon=-100.,llcrnrlat=24.5,urcrnrlon=-82.5,urcrnrlat=37.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-91.5,area_thresh=1000.,ax=ax)

   elif str.upper(domain) == 'LOUISIANA':
      draw_counties=True
    # m = Basemap(llcrnrlon=-95.,llcrnrlat=26.,urcrnrlon=-87.5,urcrnrlat=34.,\
      m = Basemap(llcrnrlon=-95.,llcrnrlat=26.,urcrnrlon=-85,urcrnrlat=34.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-90.,area_thresh=1000.,ax=ax)
              #   lat_1=25.,lat_2=46.,lon_0=-92.5,area_thresh=1000.,ax=ax)

   elif domain == 'SRxSE':
      m = Basemap(llcrnrlon=-105,llcrnrlat=22,urcrnrlon=-70,urcrnrlat=40.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-90.,area_thresh=1000.,ax=ax)

   elif domain == 'MIDATLxNE':
      m = Basemap(llcrnrlon=-85,llcrnrlat=35.,urcrnrlon=-65,urcrnrlat=48.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-77.5,area_thresh=1000.,ax=ax)

   elif domain == 'MIDATL':
      m = Basemap(llcrnrlon=-90.,llcrnrlat=32.5,urcrnrlon=-70.,urcrnrlat=45.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-82.5,area_thresh=1000.,ax=ax)

   elif domain == 'DC-NYC':
      m = Basemap(llcrnrlon=-80.,llcrnrlat=36.,urcrnrlon=-71.,urcrnrlat=42.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-75.5,area_thresh=1000.,ax=ax)

   elif domain == 'BOX-NYC':
      m = Basemap(llcrnrlon=-77.5,llcrnrlat=38.,urcrnrlon=-67.5,urcrnrlat=45.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-73.,area_thresh=1000.,ax=ax)

   elif domain == 'Florence':
      m = Basemap(llcrnrlon=-85.,llcrnrlat=31.,urcrnrlon=-70.,urcrnrlat=40.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-77.5,area_thresh=1000.,ax=ax)

   elif domain == 'OV':
      m = Basemap(llcrnrlon=-97,llcrnrlat=34.,urcrnrlon=-83.,urcrnrlat=43.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-90.,area_thresh=1000.,ax=ax)

   elif domain == 'MIDSOUTH':
    # m = Basemap(llcrnrlon=-97.5,llcrnrlat=30.,urcrnrlon=-80.,urcrnrlat=40.,\
    # m = Basemap(llcrnrlon=-100.,llcrnrlat=27.5,urcrnrlon=-75.,urcrnrlat=42.5,\    # wider view
    # m = Basemap(llcrnrlon=-102.5,llcrnrlat=27.5,urcrnrlon=-75.,urcrnrlat=44.,\    # Barry view
      m = Basemap(llcrnrlon=-95.,llcrnrlat=30.,urcrnrlon=-79.,urcrnrlat=40.5,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-92.5,area_thresh=1000.,ax=ax)

   elif domain == 'MIDSOUTHzoom':
    # m = Basemap(llcrnrlon=-92.5,llcrnrlat=33.,urcrnrlon=-85.,urcrnrlat=39.,\      # more western view
      m = Basemap(llcrnrlon=-90.,llcrnrlat=34.,urcrnrlon=-84.,urcrnrlat=38.5,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-87.,area_thresh=1000.,ax=ax)
    #             lat_1=25.,lat_2=46.,lon_0=-87.5,area_thresh=1000.,ax=ax)

   elif domain == 'scCONUS':
      m = Basemap(llcrnrlon=-115.,llcrnrlat=25.,urcrnrlon=-80.,urcrnrlat=45.,\
    # m = Basemap(llcrnrlon=-105.,llcrnrlat=30.,urcrnrlon=-82.5,urcrnrlat=46.,\
    # m = Basemap(llcrnrlon=-105.,llcrnrlat=30.,urcrnrlon=-87.5,urcrnrlat=42.5,\
                  resolution='i',projection='lcc',\
    #             lat_1=25.,lat_2=46.,lon_0=-95,area_thresh=1000.,ax=ax)
                  lat_1=25.,lat_2=46.,lon_0=-97.5,area_thresh=1000.,ax=ax)

   elif domain == 'cCONUS':
      m = Basemap(llcrnrlon=-110.,llcrnrlat=30.,urcrnrlon=-85.,urcrnrlat=46.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-100,area_thresh=1000.,ax=ax)

   elif domain == 'CP':
      m = Basemap(llcrnrlon=-105.,llcrnrlat=35.,urcrnrlon=-87.5,urcrnrlat=45.,\
    # m = Basemap(llcrnrlon=-105.,llcrnrlat=30.,urcrnrlon=-87.5,urcrnrlat=42.5,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-97.5,area_thresh=1000.,ax=ax)

   elif domain == 'CPL':
      m = Basemap(llcrnrlon=-105.,llcrnrlat=32.5,urcrnrlon=-87.5,urcrnrlat=45.,\
    # m = Basemap(llcrnrlon=-105.,llcrnrlat=30.,urcrnrlon=-87.5,urcrnrlat=42.5,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-97.5,area_thresh=1000.,ax=ax)

   elif domain == 'SPL':
    # m = Basemap(llcrnrlon=-105.,llcrnrlat=25.,urcrnrlon=-87.5,urcrnrlat=40.,\
      m = Basemap(llcrnrlon=-107.5,llcrnrlat=30.,urcrnrlon=-90.,urcrnrlat=40.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-97.5,area_thresh=1000.,ax=ax)

   elif domain == 'OK':
      m = Basemap(llcrnrlon=-104.,llcrnrlat=31.5,urcrnrlon=-92.5,urcrnrlat=39.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-97.5,area_thresh=1000.,ax=ax)

   elif domain == 'WA-BC':
      m = Basemap(llcrnrlon=-132.5,llcrnrlat=42.5,urcrnrlon=-117.5,urcrnrlat=52.5,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-125.,area_thresh=1000.,ax=ax)

   elif domain == 'Seattle':
      m = Basemap(llcrnrlon=-127.5,llcrnrlat=42.5,urcrnrlon=-117.5,urcrnrlat=50.5,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-122.5,area_thresh=1000.,ax=ax)

   elif str.upper(domain) == 'AUSTRALIA':
      m = Basemap(llcrnrlon=104.,llcrnrlat=-45.,urcrnrlon=159.,urcrnrlat=5.,\
                  resolution='i',projection='merc',\
                  lat_ts=-20.,area_thresh=1000.,ax=ax)
                 #resolution='i',projection='lcc',\
                 #lat_1=-25.,lat_2=-46.,lon_0=130.,area_thresh=1000.,ax=ax)

   m.drawcoastlines()
   m.drawstates(linewidth=0.75)
   m.drawcountries()


      parallels = np.arange(-90.,91.,latlongrid)
      m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
      meridians = np.arange(0.,360.,latlongrid)
      m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
   else:
      m.drawcounties(zorder=10.)
      latlongrid = 5.
   '''
   
   barb_length = 5.5
   if domain in large_domains or domain in regional_domains:
      latlongrid = 10.
      if domain in large_domains:
          barb_length = 4.5

   if str.upper(model_str[0:3]) == 'GFS' or str.upper(model_str[0:2]) == 'EC':
      HLwindow = 25
      HLfont = 15
      if domain in large_domains:
         skip = 10
      elif domain in regional_domains:
         skip = 7
         skip = 6
      elif domain in subregional_domains:
         skip = 4
      elif domain in state_domains:
         skip = 2

   elif str.upper(model_str) == 'RAP' or str.upper(model_str) == 'NAM' or str.upper(model_str) == 'AQM':
      HLwindow = 50
      HLfont = 15
      if domain in large_domains:
         skip = 15
      elif domain in regional_domains:
         skip = 12
      elif domain in subregional_domains:
         skip = 7
      elif domain in state_domains:
         skip = 4

   elif str.upper(model_str[0:6]) == 'HIRESW':
      HLwindow = 250
      HLfont = 15
      if domain in large_domains:
         skip = 30
      elif domain in regional_domains:
         skip = 28
      elif domain in subregional_domains:
         skip = 15
      elif domain in state_domains:
         skip = 10

   elif str.upper(model_str) == 'NAM3' or str.upper(model_str) == 'HRRR':
      HLwindow = 300
      HLfont = 15
      if domain in large_domains:
         skip = 50
      elif domain in regional_domains:
         skip = 40
      elif domain in subregional_domains:
         skip = 25
      elif domain in state_domains:
         skip = 12

   time2 = time.perf_counter()
   time3 = round(time2-time1, 3)
   print(('%.3f seconds to plot domain: ') % time3)

   time1 = time.perf_counter()

   # Temperature
   if field[0:2] == 't2':
      clevs = np.arange(-16,132,4)
      tlevs = [str(clev) for clev in clevs]

      colormap = cmap_t2m()
#     colormap_r = colormap.reversed()
      norm = matplotlib.colors.BoundaryNorm(clevs, colormap.N)

      fill_var = t2
      print(fill_var.shape)

      fill = ax.contourf(lons,lats,fill_var,clevs,cmap=colormap,norm=norm,transform=transform,extend='both')

    # cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.5,aspect=15)
      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.65,aspect=20)
#     cbar.ax.set_xticklabels(tlevs)
#     cbar.ax.tick_params(labelsize=10)
      cbar.ax.tick_params(labelsize=6)
      cbar.set_label('\xb0''F')

    # for label in cbar.ax.xaxis.get_ticklabels():
    #    label.set_visible(False)
    # for label in cbar.ax.xaxis.get_ticklabels()[::4]:
    #    label.set_visible(True)

      field_title = '2-m Temperature'

      time2 = time.perf_counter()
      time3 = round(time2-time1, 3)
      print(('%.3f seconds to plot 2-m T: ') % time3)


      try:
         if str.lower(field[3:10]) == '10mwind':

            u_var = u10
            v_var = v10

            ax.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],length=barb_length,sizes={'spacing':0.2},pivot='middle',transform=transform)

            field_title = '2-m Temperature and 10-m Wind'

      except IndexError:
         sys.exc_clear()


   # Dewpoint
   elif field[0:3] == 'td2':
      clevs = np.arange(-10,81,2)
      tlevs = [str(clev) for clev in clevs]

#     colors1 = plt.cm.gist_earth_r(np.linspace(0,0.15,40))
#     colors2 = plt.cm.PRGn(np.linspace(0.7,1,10))
      colors1 = plt.cm.terrain(np.linspace(0.75,0.92,25))
      colors2 = plt.cm.PRGn(np.linspace(0.65,1,10))
      colors3 = plt.cm.BrBG(np.linspace(0.8,0.95,5))
      colors4 = plt.cm.PuOr(np.linspace(0.8,0.9,5))
      newcolors = np.vstack((colors1,colors2,colors3,colors4))

      colormap = matplotlib.colors.ListedColormap(newcolors)
      norm = matplotlib.colors.BoundaryNorm(clevs, colormap.N)

      fill_var = td2

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,cmap=colormap,norm=norm,extend='both')

    # cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.5,aspect=15)
      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.set_xticklabels(tlevs)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label(u'\xb0''F')

      for label in cbar.ax.xaxis.get_ticklabels():
         label.set_visible(False)
      for label in cbar.ax.xaxis.get_ticklabels()[::5]:
         label.set_visible(True)

      field_title = '2-m Dewpoint'

      try:
         if str.lower(field[4:11]) == '10mwind':

            u_var = u10
            v_var = v10

            m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle')

            field_title = '2-m Dewpoint and 10-m Wind'

      except IndexError:
         sys.exc_clear()



   # 10-m Wind
   elif field == '10mwind':

       clevs = [10,15,20,25,30,40,50]
       colorlist = ['lightsteelblue','skyblue','deepskyblue','dodgerblue','lightpink','fuchsia','darkmagenta']

       fill_var = isotach_10m
       fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

       cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
       cbar.ax.tick_params(labelsize=10)
       cbar.set_label('kts')

       u_var = u10
       v_var = v10

       m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle')

       field_title = '10-m Wind'


   # SST
   elif field == 'sst':
      clevs = np.arange(0,41,1)
      tlevs = [str(clev) for clev in clevs]
      colormap = cm.get_cmap('nipy_spectral')

      fill_var = np.ma.masked_where(land == 1,skint)

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,cmap=colormap,extend='max')

    # cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.5,aspect=15)
      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.set_xticklabels(tlevs)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('C')

      for label in cbar.ax.xaxis.get_ticklabels():
         label.set_visible(False)
      for label in cbar.ax.xaxis.get_ticklabels()[::5]:
         label.set_visible(True)

      field_title = 'Sea Surface Temperature'


   # Accumulated precip
   elif field[1:4] == 'pcp' or field[2:5] == 'pcp':
      clevs = [0.01,0.1,0.25,0.5,0.75,1,1.5,2,2.5,3,4,5,6,8,10,12,14,16,18,20]
      tlevs = [str(clev) for clev in clevs]
      colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','cyan','slateblue', \
                   'mediumorchid','darkmagenta','darkred','crimson','darkorange','salmon',  \
                   'yellow','saddlebrown','magenta','pink','beige','black']

      stride = fhrs[1] - fhrs[0]

      if field == 'apcp':
         fill_var = APCP[j]

      elif field == 'apcp24':
         if fhrs[j] < 24:
            accum_length = fhrs[j] - fhrs[0]

            if j == 0 and fhrs[j] != 0:
               APCP[j] = APCP[j]*0.

            if stride == 1 and j < 24:
               fill_var = np.sum(np.array(APCP),axis=0)
            elif stride == 3 and j < 8:
               fill_var = np.sum(np.array(APCP),axis=0)
            elif stride == 6 and j < 4:
               fill_var = np.sum(np.array(APCP),axis=0)
         
         else:
            accum_length = 24
            if j == 0 and fhrs[j] == 24:
               fill_var = APCP[j]

      #     elif fhrs[j] == 24 and str.upper(model_str) in ARWs:
      #        fill_var = rt_bucket

            elif stride == 1 and j >= 24:
               fill_var = np.sum(np.array(APCP[j-23:j+1]),axis=0)

            elif stride == 3 and j >= 8:
               fill_var = np.sum(np.array(APCP[j-7:j+1]),axis=0)

            elif stride == 6 and j >= 4:
               fill_var = np.sum(np.array(APCP[j-3:j+1]),axis=0)
            
            elif stride == 1 and j < 24:
               fill_var = np.sum(np.array(APCP),axis=0)

            elif stride == 3 and j < 8:
               fill_var = np.sum(np.array(APCP),axis=0)

            elif stride == 6 and j < 4:
               fill_var = np.sum(np.array(APCP),axis=0)
            
               


      elif field == 'acpcp':
         fill_var = acprecip
         clevs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
         tlevs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
         colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','cyan','slateblue', \
                     'mediumorchid','darkmagenta','darkred','crimson']
      elif field == 'ncpcp':
         fill_var = precip - acprecip

      if fhrs[j] == 0:
         fill_var = mslp*0.

      
      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

    # cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.5,aspect=15)
      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.set_xticklabels(tlevs)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('in')

      if field == 'apcp':
      #  field_title = str(fhrs[j]-fhrs[0])+'-h QPF'
      #  field = str(fhrs[j]-fhrs[0])+'h'+field
         field_title = str(stride)+'-h QPF'
         field = str.lower(field)+str(stride)+'h'
      elif field == 'apcp24':
         field_title = str(accum_length)+'-h QPF'
         field = str.lower(field)+'h'
      #  field = str.lower(field[0:4])+str(accum_length)+'h'
      elif field == 'acpcp':
      #  field_title = str(fhrs[j]-fhrs[0])+'-h Convective Precipitation'
      #  field = str(fhrs[j]-fhrs[0])+'h'+field
         field_title = accum_str+' Convective Precipitation'
      elif field == 'ncpcp':
      #  field_title = str(fhrs[j]-fhrs[0])+'-h Gridscale Precipitation'
      #  field = str(fhrs[j]-fhrs[0])+'h'+field
         field_title = accum_str+' Gridscale Precipitation'
    # field = accum_str[0]+'h-'+field


   # Snowfall
   elif field == 'snod' or field == 'weasd':
      clevs = [0.1,1,2,3,4,5,6,8,12,18,24,30,36,48]
      tlevs = [str(clev) for clev in clevs]
      colorlist = ['powderblue','lightsteelblue','cornflowerblue','steelblue', \
                   'royalblue','blue', \
                   'khaki','orange','darkorange','red', \
                   'firebrick','darkred','maroon','indigo']

      if field == 'snod' and (fhrs[j]-fhrs[0]) > 0:
         fill_var = snod
      elif field == 'weasd' and (fhrs[j]-fhrs[0]) > 0:
         fill_var = weasd

      if (fhrs[j]-fhrs[0]) == 0:
         fill_var = mslp*0.

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

    # cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.5,aspect=15)
      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.set_xticklabels(tlevs)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('in')

      if field == 'snod':
         field_title = str(fhrs[j]-fhrs[0])+'-h Snow Depth'
         field = str(fhrs[j]-fhrs[0])+'h'+field
      elif field == 'weasd':
         field_title = str(fhrs[j]-fhrs[0])+'-h Snow Water Equivalent (10:1)'
         field = str(fhrs[j]-fhrs[0])+'h'+field


   # SLP
   elif str.lower(field) == 'slp':
      print('plotting '+str.upper(model_str)+' '+str.upper(field))

      var = scipy.ndimage.gaussian_filter(mslp,2)
      if str.upper(model_str[0:4]) in HiResModels:
         var = scipy.ndimage.gaussian_filter(var,2)

      clevs = np.arange(940,1045,4)
      cint  = np.arange(940,1045,4)
      tlevs = [str(clev) for clev in clevs]
      colormap = ncepy.ncl_perc_11Lev()
      norm = matplotlib.colors.BoundaryNorm(clevs, colormap.N)

      fill_var = mslp
    # fill_var = pmsl
      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,cmap=colormap,norm=norm,extend='both')

      contour_var = mslp
    # contour_var = pmsl
      contours = m.contour(lons,lats,contour_var,cint,colors='k',linestyles='solid',latlon=True)
      plt.clabel(contours,cint,colors='k',inline=1,fmt='%.0f',fontsize=9)
      dawsonpy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow,font=HLfont)
    # ncepy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow)
    # ncepy.plt_highs_and_lows(m,pmsl,lons,lats,mode='reflect',window=HLwindow)

    # cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.5,aspect=15)
      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.set_xticklabels(tlevs)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('mb')

      for label in cbar.ax.xaxis.get_ticklabels():
         label.set_visible(False)
      for label in cbar.ax.xaxis.get_ticklabels()[::4]:
         label.set_visible(True)

      field_title = 'SLP'
    # field_title = 'SLP (PMSL)'


   # SLP and other fields
   elif str.lower(field[0:3]) == 'slp' and len(field) > 3:
      print('plotting '+str.upper(model_str)+' '+str.upper(field))

      try:
         if str.lower(field[4:11]) == '10mwind':
            print('plotting '+str.upper(model_str)+' '+str.upper(field[4:11]))

            u_var = u10
            v_var = v10

            # do color shade winds if slp_10mwind
            if field == 'slp_10mwind':
               fill_var = sqrt(u_var**2+v_var**2)
             # clevs = np.linspace(30,140,12)
               clevs = np.linspace(10,140,14)
               colormap = ncepy.ncl_perc_11Lev()
               units = 'kts'
               field_title = 'SLP and 10-m Wind'

            elif field == 'slp_10mwind_pw':
               fill_var = pw
               clevs = [0.25,0.5,0.75,1,1.25,1.5,1.75,2.,2.25,2.5,2.75,3.,3.25,3.5]
               colormap = cm.get_cmap('gist_earth_r')
               units = 'in'
               field_title = 'SLP, PW, and 10-m Wind Barbs (kts)'

            fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,cmap=colormap,extend='max')
            cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label(units)

            m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle')


         elif str.lower(field[4:10]) == 'rh_avg':
            print('plotting '+str.upper(model_str)+' '+str.upper(field[4:10]))

            fill_var = rh_avg
            clevs = np.arange(0,101,2)
            colormap = cm.get_cmap('gist_earth_r')
            units = 'in'

            fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,cmap=colormap)
            cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('%')

            for label in cbar.ax.xaxis.get_ticklabels():
               label.set_visible(False)
            for label in cbar.ax.xaxis.get_ticklabels()[::5]:
               label.set_visible(True)

            u_shr = uwind_250 - uwind_850
            v_shr = vwind_250 - vwind_850
          # vectors = m.quiver(lons[::skip,::skip],lats[::skip,::skip],u_shr[::skip,::skip],v_shr[::skip,::skip],latlon=True,scale=700)

            field_title = 'SLP and 850-250 mb Average Relative Humidity'


         elif str.lower(field[4:8]) == 'z500':
            print('plotting '+str.upper(model_str)+' '+str.upper(field[4:8]))

            fill_var = hghts_500
            contour_var = hghts_500
            u_var = uwind_500
            v_var = vwind_500

            clevs = np.arange(492.,596.,6.)
            colormap = ncepy.ncl_perc_11Lev()
            cint = np.arange(498.,651.,6.)

            fill_var = scipy.ndimage.gaussian_filter(fill_var,1)
            contour_var = scipy.ndimage.gaussian_filter(contour_var,1)
            if str.upper(model_str[0:4]) in HiResModels:
               fill_var = scipy.ndimage.gaussian_filter(fill_var,1)
               contour_var = scipy.ndimage.gaussian_filter(contour_var,1)

            fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,cmap=colormap,extend='both')

            contours = m.contour(lons,lats,contour_var,cint,colors='k',linewidths=1.5,latlon=True)
            ax.clabel(contours,cint,colors='k',inline=1,fmt='%.0f',fontsize=10)

            m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle',color='steelblue')

            cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('dam')


#     for label in cbar.ax.xaxis.get_ticklabels():
#        label.set_visible(False)
#     for label in cbar.ax.xaxis.get_ticklabels()[::5]:
#        label.set_visible(True)

            field_title = 'SLP and '+field[5:]+' mb Heights and Winds (kts)'


         elif str.lower(field[6:10]) == 'cape':
            print('plotting '+str.upper(model_str)+' '+str.upper(field))


            clevs = [100,250,500,750,1000,1250,1500,1750,2000,2500,3000,3500,4000,5000]
            colorlist = ['lightgray','silver','darkgray','gray', \
                         'lightblue','skyblue','cornflowerblue','steelblue', \
                         'chartreuse','limegreen','yellow','gold','darkorange','red']

            if str.lower(field[4:6]) == 'sb':
               fill_var = sbcape
            elif str.lower(field[4:6]) == 'ml':
               fill_var = mlcape
            elif str.lower(field[4:6]) == 'mu':
               fill_var = mucape

            fill_var = scipy.ndimage.gaussian_filter(fill_var,1)
            if str.upper(model_str[0:4]) in HiResModels:
               fill_var = scipy.ndimage.gaussian_filter(fill_var,1)

            fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')
            cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('J $\mathregular{kg^{-1}}$')

            field_title = 'SLP and '+str.upper(field[4:10])

            try:
               if str.lower(field[11:16]) == 'shear':
                  print('plotting '+str.upper(model_str)+' '+str.upper(field[11:18])+' analysis')

                  if field[16] == '0' and field[17] == '6':
                     u_var = u_shr06
                     v_var = v_shr06
                  elif field[16] == '0' and field[17] == '1':
                     u_var = u_shr01
                     v_var = v_shr01

                  m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle')

                  field_title = 'SLP, '+str.upper(field[4:10])+', and '+field[16]+'-'+field[17]+' km '+str.title(field[11:16])+' Vectors (kts)'

            except IndexError:
               sys.exc_clear()


      except IndexError:
         sys.exc_clear()


      contour_var = mslp
      contour_var = scipy.ndimage.gaussian_filter(contour_var,2)
      if str.upper(model_str[0:4]) in HiResModels:
         contour_var = scipy.ndimage.gaussian_filter(contour_var,2)

      cint  = np.arange(900.,1100.,4.)
      contours = m.contour(lons,lats,contour_var,cint,colors='k',linestyles='solid',latlon=True)
      plt.clabel(contours,cint,colors='k',inline=1,fmt='%.0f',fontsize=9)
    # ncepy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow)
      dawsonpy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow,font=HLfont)



   # CAPE / CIN / SHEAR
   elif str.lower(field[2:6]) == 'cape':
      print('plotting '+str.upper(model_str)+' '+str.upper(field[0:6]))

      clevs = [100,250,500,750,1000,1250,1500,1750,2000,2500,3000,3500,4000,5000]
      colorlist = ['lightgray','silver','darkgray','gray', \
                   'lightblue','skyblue','cornflowerblue','steelblue', \
                   'chartreuse','limegreen','yellow','gold','darkorange','red']

      if str.lower(field[0:2]) == 'sb':
         fill_var = sbcape
      elif str.lower(field[0:2]) == 'ml':
         fill_var = mlcape
      elif str.lower(field[0:2]) == 'mu':
         fill_var = mucape

      fill_var = scipy.ndimage.gaussian_filter(fill_var,1)
      if str.upper(model_str[0:4]) in HiResModels:
         fill_var = scipy.ndimage.gaussian_filter(fill_var,1)

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('J $\mathregular{kg^{-1}}$')

      field_title = str.upper(field[0:6])

      try:
         if str.lower(field[9:12]) == 'cin':
            print('plotting '+str.upper(model_str)+' '+str.upper(field[7:12])+' analysis')
 
            if str.lower(field[0:2]) == 'sb':
               contour_var = sbcin
            elif str.lower(field[0:2]) == 'ml':
               contour_var = mlcin
      
            contour_var = scipy.ndimage.gaussian_filter(contour_var,2)
            if str.upper(model_str[0:4]) in HiResModels:
               contour_var = scipy.ndimage.gaussian_filter(contour_var,2)
   

            cint = [-100.,-25.]
            contours = m.contour(lons,lats,contour_var,cint,colors='k',linewidths=0.5,linestyles='solid',latlon=True)
            plt.clabel(contours,cint,colors='k',inline=1,fmt='%.0f',fontsize=9)

            hatch = m.contourf(lons,lats,contour_var,cint,extend='min',colors='none',hatches=['\/\/','..'],latlon=True)

            field_title = str.upper(field[0:6])+' and '+str.upper(field[7:12])

      except IndexError:
         sys.exc_clear()

      try:
         if str.lower(field[13:18]) == 'shear':
            print('plotting '+str.upper(model_str)+' '+str.upper(field[13:20])+' analysis')

            if field[18] == '0' and field[19] == '6':
               u_var = u_shr06
               v_var = v_shr06
            elif field[18] == '0' and field[19] == '1':
               u_var = u_shr01
               v_var = v_shr01

            m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle')

            field_title = str.upper(field[0:6])+', '+str.upper(field[7:12])+', and '+field[18]+'-'+field[19]+' km '+str.title(field[13:18])

      except IndexError:
         sys.exc_clear()


   # Lapse Rate
   elif str.lower(field[0:2]) == 'lr':
      print('plotting '+str.upper(model_str)+' '+str.upper(field))

      colorlist = ['lightgray','silver','darkgray','gray', \
                   'lightblue','skyblue','cornflowerblue','steelblue', \
                   'chartreuse','limegreen','yellow','gold','darkorange','red','magenta']

      if str.lower(field[2:4]) == '75':
         clevs = [5,6,7,7.25,7.5,7.75,8,8.25,8.5,8.75,9,9.25,9.5,9.75,10]
         fill_var = lr75
         field_title = '700 mb - 500 mb Lapse Rate'

      tlevs = [str(clev) for clev in clevs]

#     fill_var = scipy.ndimage.gaussian_filter(fill_var,1)
#     if str.upper(model_str[0:4]) in HiResModels:
#        fill_var = scipy.ndimage.gaussian_filter(fill_var,1)

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='both')
      fill.cmap.set_over('magenta')

      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.set_xticklabels(tlevs)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label(u'\xb0''C $\mathregular{km^{-1}}$')

   #  for label in cbar.ax.xaxis.get_ticklabels():
   #     label.set_visible(False)
   #  for label in cbar.ax.xaxis.get_ticklabels()[::5]:
   #     label.set_visible(True)


   # PBL Height
   elif str.lower(field) == 'hpbl':

      colorlist = ['lightgray','silver','darkgray','gray', \
                   'lightblue','skyblue','cornflowerblue','steelblue', \
                   'chartreuse','limegreen','yellow','gold','darkorange','red','magenta']

      clevs = [50,100,250,500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
      tlevs = [str(clev) for clev in clevs]
      fill_var = hpbl
      field_title = 'PBL Height'

      fill = ax.contourf(lons,lats,fill_var,clevs,colors=colorlist,transform=transform,extend='both')
      fill.cmap.set_over('magenta')

      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.75,aspect=20)
      cbar.ax.set_xticklabels(tlevs)
      cbar.ax.tick_params(labelsize=8)
      cbar.set_label('m')


   # Isotachs
   elif field[0:4] == 'wind':

      if field == 'wind500':
         fill_var = isotach_500
         contour_var = hghts_500
         u_var = uwind_500
         v_var = vwind_500

         clevs = [50,60,70,80,90,100,120]
         clevs = [30,40,50,60,80,100,120]
         colorlist = ['lightsteelblue','skyblue','deepskyblue','dodgerblue','lightpink','fuchsia','darkmagenta']
         cint = np.arange(498.,651.,6.)


    # fill_var = scipy.ndimage.gaussian_filter(fill_var,1)
      contour_var = scipy.ndimage.gaussian_filter(contour_var,1)
      if str.upper(model_str[0:4]) in HiResModels:
    #    fill_var = scipy.ndimage.gaussian_filter(fill_var,1)
         contour_var = scipy.ndimage.gaussian_filter(contour_var,1)


      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

      contours = m.contour(lons,lats,contour_var,cint,colors='k',linewidths=1.5,latlon=True)
      ax.clabel(contours,colors='k',inline=1,fmt='%.0f',fontsize=10)

      m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle')

      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('kts')

  #   ncepy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow)
      dawsonpy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow,font=HLfont)

#     for label in cbar.ax.xaxis.get_ticklabels():
#        label.set_visible(False)
#     for label in cbar.ax.xaxis.get_ticklabels()[::5]:
#        label.set_visible(True)

      field_title = field[4:]+' mb Heights, Winds, and Surface Lows/Highs'


   # Vorticity
   elif field[0:7] == 'vort500':

      if field[0:7] == 'vort500':
         fill_var = vort_500
         contour_var = hghts_500
         u_var = uwind_500
         v_var = vwind_500

         clevs = [16,20,24,28,32,36,40]
         colorlist = ['yellow','gold','goldenrod','orange','orangered','red','darkred']
         cint = np.arange(498.,651.,6.)


      contour_var = scipy.ndimage.gaussian_filter(contour_var,1)
      if str.upper(model_str[0:4]) in HiResModels:
         fill_var = scipy.ndimage.gaussian_filter(fill_var,1)
         contour_var = scipy.ndimage.gaussian_filter(contour_var,1)


      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

      contours = m.contour(lons,lats,contour_var,cint,colors='k',linewidths=1.5,latlon=True)
      ax.clabel(contours,colors='k',inline=1,fmt='%.0f',fontsize=10)

      m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle',color='steelblue')

      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('$\mathregular{x10^{-5}}$ $\mathregular{s^{-1}}$')

   #  ncepy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow)
      dawsonpy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow,font=HLfont)

#     for label in cbar.ax.xaxis.get_ticklabels():
#        label.set_visible(False)
#     for label in cbar.ax.xaxis.get_ticklabels()[::5]:
#        label.set_visible(True)

      field_title = field[4:]+' mb Heights, Vorticity, Winds, and Surface Lows/Highs'


   # 500 mb heights and SLP
   elif str.lower(field[0:11]) == '500hght_slp':

      fill_var = hghts_500
      clevs = np.arange(492.,596.,6.)  # 500 heights
      tlevs = np.arange(492.,596.,12.)  # 500 heights
      colormap = ncepy.ncl_perc_11Lev()

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,cmap=colormap,extend='both')
      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('dam')

      contour_var = mslp
      contour_var = scipy.ndimage.gaussian_filter(contour_var,2)
      if str.upper(model_str[0:4]) in HiResModels:
         contour_var = scipy.ndimage.gaussian_filter(contour_var,2)

      cint  = np.arange(900.,1100.,4.)
      contours = m.contour(lons,lats,contour_var,cint,colors='k',linestyles='solid',latlon=True)
      ax.clabel(contours,colors='k',inline=1,fmt='%.0f',fontsize=9)
    # ax.clabel(contours,cint,colors='k',inline=1,fmt='%d',fontsize=10)
    # ncepy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow)
      dawsonpy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow,font=HLfont)

      field_title = '500 mb Heights and SLP'

      try:
         if str.lower(field[-7:]) == '10mwind':

            u_var = u10
            v_var = v10

            m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle')

            field_title = '500 mb Heights, SLP, and 10-m Wind'

      except IndexError:
         sys.exc_clear()



   # RH
   elif field[0:2] == 'rh':
    # clevs = [70,90,100]
    # tlevs = [70,90]
    # colorlist = ['greenyellow','limegreen']
      clevs = np.arange(0,101,2)
      colormap = cm.get_cmap('gist_earth_r')

      if field == 'rh800':
         fill_var = np.ma.masked_where(psfc < 800, rh_800)
         u_var = np.ma.masked_where(psfc < 800, uwind_800)
         v_var = np.ma.masked_where(psfc < 800, vwind_800)
      elif field == 'rh825':
         fill_var = np.ma.masked_where(psfc < 825, rh_825)
         u_var = np.ma.masked_where(psfc < 825, uwind_825)
         v_var = np.ma.masked_where(psfc < 825, vwind_825)
      elif field == 'rh850':
         fill_var = np.ma.masked_where(psfc < 850, rh_850)
         u_var = np.ma.masked_where(psfc < 850, uwind_850)
         v_var = np.ma.masked_where(psfc < 850, vwind_850)
      elif field == 'rh875':
         fill_var = np.ma.masked_where(psfc < 875, rh_875)
         u_var = np.ma.masked_where(psfc < 875, uwind_875)
         v_var = np.ma.masked_where(psfc < 875, vwind_875)
      elif field == 'rh900':
         fill_var = np.ma.masked_where(psfc < 900, rh_900)
         u_var = np.ma.masked_where(psfc < 900, uwind_900)
         v_var = np.ma.masked_where(psfc < 900, vwind_900)
       # fill_var = rh_900
       # u_var = uwind_900
       # v_var = vwind_900

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,cmap=colormap,extend='max')

#     cint = np.arange(265.,366.,5.)
#     contours = m.contour(lons,lats,hghts_700,cint,colors='k',linewidths=1.5,latlon=True)
#     ax.clabel(contours,cint,colors='k',inline=1,fmt='%.0f',fontsize=10)

#     contour_var = scipy.ndimage.gaussian_filter(contour_var,2)
#     if str.upper(model_str[0:4]) in HiResModels:
#        contour_var = scipy.ndimage.gaussian_filter(contour_var,2)

      cint = np.arange(-10.,11.,1.)
   #  contours = m.contour(lons,lats,contour_var,cint,colors='orange',latlon=True)
   #  plt.clabel(contours,cint,colors='k',inline=1,fmt='%.0f',fontsize=9)
#     ncepy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow)

  #   if cint[-1] <= 0.:
  #      contours = m.contour(lons,lats,omega_700,cint,colors='orange',linewidths=1,linestyles='dashed',latlon=True)
  #   else:
  #   contours = m.contour(lons,lats,omega_700,cint,colors='orange',linewidths=1,latlon=True)

      m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle')

      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('%')

      for label in cbar.ax.xaxis.get_ticklabels():
         label.set_visible(False)
      for label in cbar.ax.xaxis.get_ticklabels()[::5]:
         label.set_visible(True)

      field_title = field[2:5]+' mb RH and Wind Barbs'


 # Omega
   elif field[0:5] == 'omega':
    # clevs = [70,90,100]
    # tlevs = [70,90]
    # colorlist = ['greenyellow','limegreen']
      clevs = np.arange(-60,61,2)
      colormap = cm.get_cmap('seismic_r')

      if field == 'omega':
         fill_var = omega
         contour_var = hghts_700
      elif field == 'omega700':
         fill_var = np.ma.masked_where(psfc < 700, omega_700)
      #  u_var = np.ma.masked_where(psfc < 700, uwind_700)
      #  v_var = np.ma.masked_where(psfc < 700, vwind_700)
         contour_var = hghts_700
      elif field == 'omega800':
         fill_var = np.ma.masked_where(psfc < 800, omega_800)
         u_var = np.ma.masked_where(psfc < 800, uwind_800)
         v_var = np.ma.masked_where(psfc < 800, vwind_800)
         contour_var = hghts_800
      elif field == 'omega825':
         fill_var = np.ma.masked_where(psfc < 825, omega_825)
         u_var = np.ma.masked_where(psfc < 825, uwind_825)
         v_var = np.ma.masked_where(psfc < 825, vwind_825)
         contour_var = hghts_825
      elif field == 'omega850':
         fill_var = np.ma.masked_where(psfc < 850, omega_850)
         u_var = np.ma.masked_where(psfc < 850, uwind_850)
         v_var = np.ma.masked_where(psfc < 850, vwind_850)
         contour_var = hghts_850
      elif field == 'omega875':
         fill_var = np.ma.masked_where(psfc < 875, omega_875)
         u_var = np.ma.masked_where(psfc < 875, uwind_875)
         v_var = np.ma.masked_where(psfc < 875, vwind_875)
         contour_var = hghts_875
      elif field == 'omega900':
         fill_var = np.ma.masked_where(psfc < 900, omega_900)
         u_var = np.ma.masked_where(psfc < 900, uwind_900)
         v_var = np.ma.masked_where(psfc < 900, vwind_900)
         contour_var = hghts_900

    # if int(field[5:]) == 700:
    #    cint = np.arange(265.,366.,5.)    # 700 heights
    # elif int(field[5:]) >= 800:
    #    cint = np.arange(60.,241.,3.)     # 800-900 heights
    # else:
      cint = np.arange(265.,366.,5.)    # 700 heights

      fill_var = scipy.ndimage.gaussian_filter(fill_var,2)
      contour_var = scipy.ndimage.gaussian_filter(contour_var,2)
      if str.upper(model_str[0:4]) in HiResModels:
         fill_var = scipy.ndimage.gaussian_filter(fill_var,2)
         fill_var = scipy.ndimage.gaussian_filter(fill_var,2)
         contour_var = scipy.ndimage.gaussian_filter(contour_var,2)
         contour_var = scipy.ndimage.gaussian_filter(contour_var,2)

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,cmap=colormap,extend='both')

      contours = m.contour(lons,lats,contour_var,cint,colors='k',linewidths=1.5,latlon=True)
      ax.clabel(contours,cint,colors='k',inline=1,fmt='%.0f',fontsize=10)

#     ncepy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow)

  #   if cint[-1] <= 0.:
  #      contours = m.contour(lons,lats,omega_700,cint,colors='orange',linewidths=1,linestyles='dashed',latlon=True)
  #   else:
  #   contours = m.contour(lons,lats,omega_700,cint,colors='orange',linewidths=1,latlon=True)

  #   m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle')

      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('$\mathregular{\mu}$$\mathregular{b}$ $\mathregular{s^{-1}}$')

      for label in cbar.ax.xaxis.get_ticklabels():
         label.set_visible(False)
      for label in cbar.ax.xaxis.get_ticklabels()[::5]:
         label.set_visible(True)

      field_title = field[5:8]+' mb Omega and Geopotential Height'
      field_title = '700 mb Geopotential Height and 700-900 mb Layer-Averaged Omega'


   # Column-integrated hydrometeors 
   elif field[0:4] == 'tcol':
      clevs = [0.001,0.005,0.01,0.05,0.1,0.25,0.5,1,2,4,6,10,15,20,25]
      colorlist = plt.cm.gist_stern_r(np.linspace(0, 1, len(clevs)+1))
      colormap = matplotlib.colors.ListedColormap(colorlist)
      norm = matplotlib.colors.BoundaryNorm(clevs,colormap.N)


      if field == 'tcolc':
         fill_var = tcolc
         field_title = 'Total Column-Integrated Condensate'
      elif field == 'tcolw':
         fill_var = tcolw
         field_title = 'Total Column-Integrated Cloud Water + Rain'
      elif field == 'tcoli':
         fill_var = tcoli
         field_title = 'Total Column-Integrated Cloud Ice + Snow'
      elif field == 'tcols':
         fill_var = tcols
         field_title = 'Total Column-Integrated Snow'
      elif field == 'tcolr':
         fill_var = tcolr
         field_title = 'Total Column-Integrated Rain'
      elif field == 'tcolci':
         fill_var = tcolci
         field_title = 'Total Column-Integrated Cloud Ice'
      elif field == 'tcolcw':
         fill_var = tcolcw
         field_title = 'Total Column-Integrated Cloud Water'

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,cmap=colormap,norm=norm,extend='both',ax=ax)

    # cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.5,aspect=15)
      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('kg m${^{-2}}$')
   #  cbar.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])

      for label in cbar.ax.xaxis.get_ticklabels():
         label.set_visible(False)
      for label in cbar.ax.xaxis.get_ticklabels()[::2]:
         label.set_visible(True)


   # Simulated reflectivity
   elif field[0:3] == 'ref':
      clevs = np.linspace(5,70,14)
      colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green', \
                   'yellow','gold','darkorange','red','firebrick','darkred','fuchsia','darkmagenta']

      if field[0:4] == 'refc':
         fill_var = refc
      elif field[0:5] == 'refd1':
         fill_var = refd1
      elif field[0:5] == 'refd4':
         fill_var = refd4
      
      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

    # cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.5,aspect=15)
      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('dBZ')

      if field[0:4] == 'refc':
         field_title = 'Composite Reflectivity'
      elif field[0:4] == 'refd':
         field_title = field[4:]+'-km AGL Reflectivity'

      try:
         if str.lower(field[5:7]) == 'uh':
            print('plotting '+str.upper(model_str)+' '+str.upper(field[5:7]))

            if fhrs[j] != 0:
               contour_var = uh25
            else:
               contour_var = refc*0.

            cint = [75.,1000.,]
            contours = m.contour(lons,lats,contour_var,cint,colors='k',linestyles='solid',linewidths=2.5,latlon=True)

            field_title = field_title+' and Updraft Helicity > '+str(int(cint[0]))+' $\mathregular{m^{2}}$ $\mathregular{s^{-2}}$'

      except IndexError:
         sys.exc_clear()


   # Updraft helicity 
   elif str.lower(field[0:2]) == 'uh':
#     clevs = [75,100,125,150,175,200,250,300,400]
#     colorlist = ['lime','limegreen','green', \
      clevs = [25,50,75,100,125,150,175,200,250,300,400]
      colorlist = ['turquoise','dodgerblue','lime','limegreen','green', \
                   'yellow','gold','darkorange','red','firebrick','fuchsia']

      if field == 'uh25':
         fill_var = uh25
      elif field == 'uh03':
         fill_var = uh03
      elif field == 'uh02':
         fill_var = uh02
      elif field == 'UH25_accum':
         fill_var = UH25_accum
      elif field == 'UH03_accum':
         fill_var = UH03_accum
      elif field == 'UH02_accum':
         fill_var = UH02_accum
      
      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('$\mathregular{m^{2}}$ $\mathregular{s^{-2}}$')

      if field[0:2] == 'uh':
         field_title = 'Max '+field[2]+'-'+field[3]+' km Updraft Helicity'
      elif field[0:2] == 'UH':
         accum_length = fhrs[j] - fhrs[0] + 1
       # field_title = str(fhrs[j])+'-h Max '+field[2]+'-'+field[3]+' km Updraft Helicity'
         field_title = str(accum_length)+'-h Max '+field[2]+'-'+field[3]+' km Updraft Helicity'

   # Vertical vorticity 
   elif str.lower(field[0:2]) == 'vv':
    # clevs = [0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015]
    # clevs = [0.0025,0.005,0.0075,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015]
      clevs = [25,50,75,80,90,100,110,120,130,140,150]
      clevs = [25,30,35,40,45, 50, 60, 70, 80, 90,100]
      colorlist = ['turquoise','dodgerblue','lime','limegreen','green', \
                   'yellow','gold','darkorange','red','firebrick','fuchsia']

      if field == 'vv02':
         fill_var = vv02
      elif field == 'vv01':
         fill_var = vv01
      elif field == 'VV02_accum':
         fill_var = VV02_accum
      elif field == 'VV01_accum':
         fill_var = VV01_accum
      
      fill_var = fill_var * 10000.
      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('$\mathregular{x10^{-4}}$ $\mathregular{s^{-1}}$')

      if field[0:2] == 'vv':
         field_title = 'Max '+field[2]+'-'+field[3]+' km Vertical Vorticity'
      elif field[0:2] == 'VV':
         accum_length = fhrs[j] - fhrs[0] + 1
       # field_title = str(fhrs[j])+'-h Max '+field[2]+'-'+field[3]+' km Vertical Vorticity'
         field_title = str(accum_length)+'-h Max '+field[2]+'-'+field[3]+' km Vertical Vorticity'


   # Soil Moisture
   elif field[0:5] == 'soilw':

      clevs = [0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5]
      tlevs = [str(clev) for clev in clevs]
      colorlist = ['saddlebrown','darkgoldenrod','darkorange','orange','#EEC900','chartreuse','limegreen','green','#1C86EE']

      fill_var = soilw0_10
    # fill_var = np.ma.masked_where(land == 0,soilw0_10)

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

    # cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.5,aspect=15)
      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.set_xticklabels(tlevs)
      cbar.ax.tick_params(labelsize=10)
   #  cbar.set_label('')

   #  for label in cbar.ax.xaxis.get_ticklabels():
   #     label.set_visible(False)
   #  for label in cbar.ax.xaxis.get_ticklabels()[::5]:
   #     label.set_visible(True)

      field_title = '0-10 cm Soil Moisture'


   # Probability
   elif field[0:4] == 'prob':

      clevs = [5,10,20,30,40,50,60,70,80,90,95]
      colorlist = ['blue','dodgerblue','cyan','limegreen','chartreuse','yellow', \
                   'orange','red','darkred','purple','orchid']
      print(field)

      if field == 'prob_refc40':
          fill_var = prob_refc40
          field_title = 'Probability of Reflectivity > 40 dBZ'

      elif field == 'prob_ffg1':
          accum_length = 1
          fill_var = prob_ffg1
          field_title = 'Probability of 1-h FFG Exceedance'
      elif field == 'prob_ffg3':
          accum_length = 3
          fill_var = prob_ffg3
          field_title = 'Probability of 3-h FFG Exceedance'
      elif field == 'prob_ffg6':
          accum_length = 6
          fill_var = prob_ffg6
          field_title = 'Probability of 6-h FFG Exceedance'

      elif field == 'prob_ari2_6h':
          accum_length = 6
          fill_var = prob_ari2_6h
          field_title = 'Probability of 6-h QPF Exceeding 2-Year ARI'
      elif field == 'prob_ari5_6h':
          accum_length = 6
          fill_var = prob_ari5_6h
          field_title = 'Probability of 6-h QPF Exceeding 5-Year ARI'
      elif field == 'prob_ari10_6h':
          accum_length = 6
          fill_var = prob_ari10_6h
          field_title = 'Probability of 6-h QPF Exceeding 10-Year ARI'
      elif field == 'prob_ari100_6h':
          accum_length = 6
          fill_var = prob_ari100_6h
          field_title = 'Probability of 6-h QPF Exceeding 100-Year ARI'

      elif field == 'prob_ari2_24h':
          accum_length = 24
          fill_var = prob_ari2_24h
          field_title = 'Probability of 24-h QPF Exceeding 2-Year ARI'
      elif field == 'prob_ari5_24h':
          accum_length = 24
          fill_var = prob_ari5_24h
          field_title = 'Probability of 24-h QPF Exceeding 5-Year ARI'
      elif field == 'prob_ari10_24h':
          accum_length = 24
          fill_var = prob_ari10_24h
          field_title = 'Probability of 24-h QPF Exceeding 10-Year ARI'
      elif field == 'prob_ari100_24h':
          accum_length = 24
          fill_var = prob_ari100_24h
          field_title = 'Probability of 24-h QPF Exceeding 100-Year ARI'

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

    # cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.5,aspect=15)
      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
#     cbar.ax.set_xticklabels(tlevs)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('%')

   #  for label in cbar.ax.xaxis.get_ticklabels():
   #     label.set_visible(False)
   #  for label in cbar.ax.xaxis.get_ticklabels()[::5]:
   #     label.set_visible(True)





# Plot point markers
   if plot_loc:
      # Plot bufr location
      x, y = m(olons, olats)
      m.scatter(x, y, marker='*', color='r',s=75)



   #################################
   ## TITLE AND FILENAME SETTINGS ##
   #################################

   if str.upper(model_str) == 'NAM3':
       mod_str = str.upper(model_str[0:3])+' Nest'
   elif str.upper(model_str[0:2]) == 'EC':
       mod_str = 'ECMWF'
   elif str.upper(model_str[0:4]) == 'GFSV':
       mod_str = 'GFSv'+model_str[4:]
   elif str.upper(model_str) == 'FV3GFSTEST':
       mod_str = model_str[0:6]+' Experiment'
   elif str.upper(model_str[0:6]) == 'HIRESW':
       mod_str = model_str[0:6]+' '+str.upper(model_str[6:])
   elif str.upper(model_str[4:]) == 'MEAN':
       mod_str = str.upper(model_str[0:4])+' Mean'
   elif str.upper(model_str[4:]) == 'PMMN':
       mod_str = str.upper(model_str[0:4])+' Probability-Matched Mean'
   elif str.upper(model_str[4:]) == 'AVRG':
       mod_str = str.upper(model_str[0:4])+' Blended Mean'
   elif str.upper(model_str[4:]) == 'PROB' or str.upper(model_str[4:]) == 'FFRI':
       mod_str = str.upper(model_str[0:4])
   else:
       mod_str = str.upper(model_str)

   if fhrs[j] == 0:
       mod_str = mod_str + ' Analysis'

   var_str = field_title
   initstr = date_str.strftime('Init: %HZ %d %b %Y')
   plt.text(0, 1.06, mod_str, horizontalalignment='left', transform=ax.transAxes, fontweight='bold')
   plt.text(0, 1.01, var_str, horizontalalignment='left', transform=ax.transAxes, fontweight='bold')

#  if fhrs[j] != 0:
   plt.text(1, 1.06, initstr, horizontalalignment='right', transform=ax.transAxes, fontweight='bold')

   if field[5:] == 'accum':
     start_time = date_list[0] + datetime.timedelta(hours=-1) 
   # validstr = start_time.strftime('Valid: %HZ %d %b to ')+date_list[j].strftime('%HZ %d %b %Y')  
     validstr = start_time.strftime('Valid: %HZ %d %b to ')+date_list[j].strftime('%HZ %d %b %Y')+' (F'+str(fhrs[j]).zfill(2)+')'

   elif field[0:8] == 'prob_ari' or field[0:8] == 'prob_ffg':
     start_time = date_list[j] + datetime.timedelta(hours=-accum_length) 
     validstr = start_time.strftime('Valid: %HZ %d %b to ')+date_list[j].strftime('%HZ %d %b %Y')+' (F'+str(fhrs[j]).zfill(2)+')'

#    validstr = date_list[0].strftime('Valid: %HZ %d %b to ')+date_list[j].strftime('%HZ %d %b %Y')  
   else:
      validstr = date_list[j].strftime('Valid: %HZ %d %b %Y')+' (F'+str(fhrs[j]).zfill(2)+')'

   plt.text(1, 1.01, validstr, horizontalalignment='right', transform=ax.transAxes, fontweight='bold')


   fname = str.lower(field)+'_'+str.lower(domain)+'_'+cycle+'_f'+str(fhrs[j]).zfill(2)

   plt.savefig(OUT_DIR+'/'+fname+'.png',bbox_inches='tight')
   plt.close()



#Loop to plot
for j in range(len(date_list)):
#for j in range(1):
   fhour = date_list[j].strftime("%H")
   print(fhrs[j])
   print(date_list[j])

   omega = []
   rh_avg = []

   #################
   # GFS
   if str.upper(model_str) == 'GFS':
      try:
         fil = '/lfs/h1/ops/prod/com/gfs/'+os.environ['gfs_ver']+'/gfs.'+cycle[0:8]+'/'+cycle[8:10]+'/atmos/'+ \
               str.lower(model_str)+'.t'+cycle[8:10]+'z.pgrb2.0p25.f'+str(fhrs[j]).zfill(3)
         grbs = grib2io.open(fil,mode='r')
      except:
         fil = DATA_DIR+'/gfs.'+cycle[0:8]+'/'+cycle[8:10]+'/atmos/'+ \
               str.lower(model_str)+'.t'+cycle[8:10]+'z.pgrb2.0p25.f'+str(fhrs[j]).zfill(3)
         grbs = grib2io.open(fil,mode='r')


   # EC
   elif str.upper(model_str[0:2]) == 'EC':
      try:
         fil = '/gpfs/hps/nco/ops/com/gfs/prod/gfs.'+cycle[0:8]+'/'+str.lower(model_str)+'.t'+ \
               cycle[8:10]+'z.pgrb2.0p25.f'+str(fhrs[j]).zfill(3)
         grbs = pygrib.open(fil)
      except:
         fil = DATA_DIR+'/ecmwf.'+cycle[0:8]+\
               '/ec.t'+cycle[8:10]+'z.0p25.f'+str(fhrs[j]).zfill(3)+'.grb'
         grbs = pygrib.open(fil)

      hghts_500  = grbs.select(name='Geopotential Height',typeOfLevel='isobaricInhPa',level=500)[0].values*0.1
      vort_500 = hghts_500*0.
#     vort_500 = grbs.select(name='Absolute vorticity',typeOfLevel='isobaricInhPa',level=500)[0].values*1.e5
      relv_500 = grbs.select(name='Vorticity (relative)',typeOfLevel='isobaricInhPa',level=500)[0].values
      uwind_500 = grbs.select(name='U component of wind',typeOfLevel='isobaricInhPa',level=500)[0].values*1.94384
      vwind_500 = grbs.select(name='V component of wind',typeOfLevel='isobaricInhPa',level=500)[0].values*1.94384
      isotach_500 = np.sqrt(uwind_500**2+vwind_500**2)



   #################
   # FV3GFS
   elif str.upper(model_str[0:6]) == 'FV3GFS':
      try:
         fil = '/com2/nam/prod/nam1.'+YYYYMMDDCC[0:8]+'/'+str.lower(model_str[0:3])+'.t'+ \
               YYYYMMDDCC[8:10]+'z.conusnest.hiresf00.tm00.grib2'
         grbs = pygrib.open(fil)
      except:
         fil = DATA_DIR+'/'+str.lower(model_str)+'.'+cycle[0:8]+'.t'+ \
               cycle[8:10]+'z.pgrb2.0p25.f'+str(fhrs[j]).zfill(3)
         grbs = pygrib.open(fil)

      skip = 10

    # grbs.seek(0)
    # for gr in grbs:
    #    print gr

    # if fhrs[j] == 96:
       # gr = grbs.message(260)
       # gr = grbs.select(name='Snow depth')[0]
       # print gr
       # snod0 = gr.values*39.3701
    #    snod0 = grbs.select(name='Snow depth')[0].values*39.3701

    # else:
       # gr = grbs.message(279)
       # print gr
       # temp = gr.values*39.3701
       # snod = temp - snod0
    #    snod = grbs.select(name='Snow depth')[0].values*39.3701
    #    snod = snod - snod0


      skint = grbs.select(name='Temperature',level=0)[0].values-273.15
      land = grbs.select(name='Land-sea mask')[0].values
      


   #################
   # NAM
   elif str.upper(model_str) == 'NAM':
      try:
         fil = '/com2/nam/prod/nam.'+cycle[0:8]+'/'+str.lower(model_str)+'.t'+ \
               cycle[8:10]+'z.awip12'+str(fhrs[j]).zfill(2)+'.tm00.grib2'
         grbs = pygrib.open(fil)
      except:
         fil = DATA_DIR+'/'+str.lower(model_str)+'.'+cycle[0:8]+'/'+ \
               str.lower(model_str)+'.t'+ \
               cycle[8:10]+'z.awphys'+str(fhrs[j]).zfill(2)+'.tm00.grib2'
#              cycle[8:10]+'z.awip12'+str(fhrs[j]).zfill(2)+'.tm00.grib2'
         grbs = pygrib.open(fil)

      HLwindow = 75
   #  skip = 10
      skip = 6

      grbs.seek(0)
      for gr in grbs:
         if gr.name == 'Total Precipitation':
            print(gr)

#     tcolcw = grbs.select(name='Total column-integrated cloud water')[0].values
#     tcolci = grbs.select(name='Total column-integrated cloud ice')[0].values
#     tcolr = grbs.select(name='Total column integrated rain')[0].values
#     tcols = grbs.select(name='Total column integrated snow')[0].values
#     tcolc = grbs.select(name='Total column-integrated condensate')[0].values
#     tcolw = tcolcw + tcolr
#     tcoli = tcolci + tcols


#     rh_800 = grbs.select(name='Relative humidity',typeOfLevel='isobaricInhPa',level=850)[0].values
#     omega_800 = grbs.select(name='Vertical velocity',typeOfLevel='isobaricInhPa',level=850)[0].values*10.
#     hghts_800  = grbs.select(name='Geopotential Height',typeOfLevel='isobaricInhPa',level=850)[0].values*0.1
#     uwind_800 = grbs.select(name='U component of wind',typeOfLevel='isobaricInhPa',level=850)[0].values
#     vwind_800 = grbs.select(name='V component of wind',typeOfLevel='isobaricInhPa',level=850)[0].values

#     psfc = grbs.select(name='Surface pressure')[0].values*0.01

#     refc = grbs.select(name='Maximum/Composite radar reflectivity')[0].values
#     refd1 = grbs.select(name='Derived radar reflectivity',typeOfLevel='heightAboveGround',level=1000)[0].values
#     refd4 = grbs.select(name='Derived radar reflectivity',typeOfLevel='heightAboveGround',level=4000)[0].values


      # Precip and Snow for NAM
#     if fhrs[j] == 0:
#        gr = grbs.message(682)
#        print gr
#        snod0 = gr.values*39.3701
#     elif fhrs[j] > 0:
#        gr = grbs.message(682)
#        print gr
#        snod = gr.values*39.3701
##       snod = snod - snod0

      '''
      if fhrs[j] == 0:
         accum_str = '0-h'
         precip = refc*0.
         acprecip = refc*0.
      elif fhrs[j] <= 36:
         accum_str = '1-h'
         if fhrs[j]%12 == 1:
            precip = grbs.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values/25.4
            acprecip = grbs.select(name='Convective precipitation (water)',lengthOfTimeRange=1)[0].values/25.4
         else:
            precip = grbs.select(name='Total Precipitation')[0].values/25.4
            acprecip = grbs.select(name='Convective precipitation (water)')[0].values/25.4

         #  tempfil = '/ptmpp2/Logan.Dawson/MEG/NAM/data/nam.'+cycle[0:8]+'.t'+ \
            tempfil = DATA_DIR+'/nam.'+cycle[0:8]+'.t'+ \
                      cycle[8:10]+'z.awip12'+str(fhrs[j-1]).zfill(2)+'.tm00.grib2'
            tempgrbs = pygrib.open(tempfil)

            temp = tempgrbs.select(name='Total Precipitation')[0].values/25.4
            precip = precip - temp

            temp = tempgrbs.select(name='Convective precipitation (water)')[0].values/25.4
            acprecip = acprecip - temp


      elif fhrs[j] > 36:
         accum_str = '3-h'

         if fhrs[j]%12 == 3:
            precip = grbs.select(name='Total Precipitation',lengthOfTimeRange=3)[0].values/25.4
            acprecip = grbs.select(name='Convective precipitation (water)',lengthOfTimeRange=3)[0].values/25.4
         else:
            precip = grbs.select(name='Total Precipitation')[0].values/25.4
            acprecip = grbs.select(name='Convective precipitation (water)')[0].values/25.4

            tempfil = DATA_DIR+'/nam.'+cycle[0:8]+'.t'+ \
                      cycle[8:10]+'z.awip12'+str(fhrs[j-1]).zfill(2)+'.tm00.grib2'
            tempgrbs = pygrib.open(tempfil)

            temp = tempgrbs.select(name='Total Precipitation')[0].values/25.4
            precip = precip - temp

            temp = tempgrbs.select(name='Convective precipitation (water)')[0].values/25.4
            acprecip = acprecip - temp

      '''


   #################
   # NAM Nest
   elif str.upper(model_str) == 'NAM3':
      try:
         fil = '/com2/nam/prod/nam.'+cycle[0:8]+'/'+str.lower(model_str[0:3])+'.t'+ \
               cycle[8:10]+'z.conusnest.hiresf'+str(fhrs[j]).zfill(2)+'.tm00.grib2'
         grbs = pygrib.open(fil)
      except:   
         fil = DATA_DIR+'/'+str.lower(model_str[0:3])+'.'+cycle[0:8]+'/'+ \
               str.lower(model_str[0:3])+'.t'+ \
               cycle[8:10]+'z.conusnest.hiresf'+str(fhrs[j]).zfill(2)+'.tm00.grib2'
         grbs = pygrib.open(fil)

      HLwindow = 300
      skip = 25
    # skip = 40

      # Snow for NAM3
#     if fhrs[j] == 0:
#        gr = grbs.message(856)
#        print gr
#        snod0 = gr.values*39.3701
#     elif fhrs[j] > 0:
#        print grb
#        gr = grbs.message(868)
#        print gr
#        snod = gr.values*39.3701
#        snod = snod - snod0


      # Precip for NAM3
      # first precip message is 3-h bucket; second is 1-h bucket
#     if (fhrs[1]-fhrs[0] == 3): 
#        precip_ind = 0
#     elif (fhrs[1]-fhrs[0] == 1): 
#        precip_ind = 1

#     bucket = grbs.select(name='Total Precipitation')[precip_ind].values/25.4
#     APCP.append(bucket)


#     try:
#        precip
#     except NameError:
#        precip = None

#     if precip is None:
#        precip = bucket
#     else:
#        precip = precip + bucket



#        accum_str = '0-h'
#     if fhrs[j] == 3:
#        precip = grb.values/25.4


#        if fhrs[j] == 3:
#           precip = grb.values/25.4
#        elif j == 0:
#           precip = grb.values/25.4
#        else:
#           bucket = grb.values/25.4
#           precip = precip + bucket




   #################
   # HiResW ARW
   elif str.upper(model_str) == 'HIRESWARW':
      try:
         fil = '/gpfs/hps/nco/ops/com/hiresw/prod/hiresw.'+cycle[0:8]+'/'+str.lower(model_str[0:6])+'.t'+ \
               cycle[8:10]+'z.'+str.lower(model_str[6:])+'_5km.f'+str(fhrs[j]).zfill(2)+'.conus.grib2'
         grbs = pygrib.open(fil)
      except:   
         fil = DATA_DIR+'/'+str.lower(model_str[0:6])+'.'+cycle[0:8]+'/'+ \
               str.lower(model_str[0:6])+'.t'+ \
               cycle[8:10]+'z.'+str.lower(model_str[6:])+'_5km.f'+str(fhrs[j]).zfill(2)+'.conus.grib2'
         grbs = pygrib.open(fil)

      HLwindow = 175
      skip = 15

      # Precip for HiResWARW
#     rt_bucket = grbs.select(name='Total Precipitation')[0].values/25.4
#     if j == 0:
#        APCP.append(rt_bucket)
#     else:
#        bucket = rt_bucket - APCP[j-1]
#        APCP.append(bucket)

#     if fhrs[j] > 0:
#        precip = grbs.select(name='Total Precipitation')[0].values/25.4    # first precip message is run-total bucket


   #################
   # HiResW NMMB
   elif str.upper(model_str) == 'HIRESWNMMB':
      try:
         fil = '/gpfs/hps/nco/ops/com/hiresw/prod/hiresw.'+cycle[0:8]+'/'+str.lower(model_str[0:6])+'.t'+ \
               cycle[8:10]+'z.'+str.lower(model_str[6:])+'_5km.f'+str(fhrs[j]).zfill(2)+'.conus.grib2'
         grbs = pygrib.open(fil)
      except:   
         fil = DATA_DIR+'/'+str.lower(model_str[0:6])+'.'+cycle[0:8]+'/'+ \
               str.lower(model_str[0:6])+'.t'+ \
               cycle[8:10]+'z.'+str.lower(model_str[6:])+'_5km.f'+str(fhrs[j]).zfill(2)+'.conus.grib2'
         grbs = pygrib.open(fil)

      HLwindow = 175
      skip = 15
   #  skip = 25




#     if fhrs[j] > 0:
#        bucket = grbs.select(name='Total Precipitation')[0].values/25.4    # first precip message is 3-h bucket; second is 1-h bucket
#        try:
#           precip
#        except NameError:
#           precip = None

#        if precip is None:
#           precip = bucket
#        else:
#           precip = precip + bucket


   #################
   # HiResW ARW2
   elif str.upper(model_str) == 'HIRESWARW2':
      try:
         fil = '/gpfs/hps/nco/ops/com/hiresw/prod/hiresw.'+cycle[0:8]+'/'+str.lower(model_str[0:6])+'.t'+ \
               cycle[8:10]+'z.'+str.lower(model_str[6:9])+'_5km.f'+str(fhrs[j]).zfill(2)+'.conusmem2.grib2'
         grbs = pygrib.open(fil)
      except:   
         fil = DATA_DIR+'/'+str.lower(model_str[0:6])+'.'+cycle[0:8]+'/'+ \
               str.lower(model_str[0:6])+'.t'+ \
               cycle[8:10]+'z.'+str.lower(model_str[6:9])+'_5km.f'+str(fhrs[j]).zfill(2)+'.conusmem2.grib2'
         grbs = pygrib.open(fil)

      HLwindow = 175
      skip = 15

      # Precip for HiResWARW2
#     if fhrs[j] > 0:
#        precip = grbs.select(name='Total Precipitation')[0].values/25.4    # first precip message is run-total bucket


   #################
   # HiResW FV3
   elif str.upper(model_str) == 'HIRESWFV3':
      try:
         fil = '/gpfs/hps/nco/ops/com/hiresw/prod/hiresw.'+cycle[0:8]+'/'+str.lower(model_str[0:6])+'.t'+ \
               cycle[8:10]+'z.'+str.lower(model_str[6:9])+'_5km.f'+str(fhrs[j]).zfill(2)+'.conus.grib2'
         grbs = pygrib.open(fil)
      except:   
         fil = DATA_DIR+'/'+str.lower(model_str[0:6])+'.'+cycle[0:8]+'/'+ \
               str.lower(model_str[0:6])+'.t'+ \
               cycle[8:10]+'z.'+str.lower(model_str[6:])+'_5km.f'+str(fhrs[j]).zfill(2)+'.conus.grib2'
         grbs = pygrib.open(fil)

      HLwindow = 175
      skip = 15

      # Precip for HiResWFV3
#     if fhrs[j] > 0:
#        precip = grbs.select(name='Total Precipitation')[0].values/25.4    # first precip message is run-total bucket

   #################
   # HREF
   elif str.upper(model_str[0:4]) == 'HREF' and fhrs[j] > 0:
      try:
         fil = '/gpfs/hps/nco/ops/com/hiresw/prod/href.'+cycle[0:8]+'/ensprod/'+str.lower(model_str[0:4])+'.t'+ \
               cycle[8:10]+'z.conus.'+str.lower(model_str[4:])+'.f'+str(fhrs[j]).zfill(2)+'.grib2'
         grbs = pygrib.open(fil)
      except:
         fil = DATA_DIR+'/'+str.lower(model_str[0:4])+'.'+cycle[0:8]+'.t'+ \
               cycle[8:10]+'z.conus.'+str.lower(model_str[4:])+'.f'+str(fhrs[j]).zfill(2)+'.grib2'
         grbs = pygrib.open(fil)

      HLwindow = 175
      skip = 25


      if str.upper(model_str) == 'HREFFFRI':
          grb = grbs.select(name='Probability of precipitation exceeding flash flood guidance values')[0]
          fields = ['prob_ffri']
      else: 
          grb = grbs.select(name='Total Precipitation')[0]

  #   grb = grbs.select(name='Maximum/Composite radar reflectivity')[3]
  #   prob_refc40 = grb.values

#     grbs.seek(0)
#     for gr in grbs:
#        if gr.name == 'Total Precipitation':
#        print gr

      # Precip for HREF AVRG
#     grb = grbs.select(name='Total Precipitation')[1]    # first precip message is 1-h bucket; second is 3-h bucket
#     bucket = grb.values/25.4

#     try:
#        precip
#     except NameError:
#        precip = None

#     if precip is None:
#        precip = bucket
#     else:
#        precip = precip + bucket


   #################
   # RAP
   elif str.upper(model_str[0:3]) == 'RAP':
      try:
         fil = '/gpfs/hps/nco/ops/com/rap/prod/rap.'+cycle[0:8]+'/'+str.lower(model_str[0:6])+'.t'+ \
               cycle[8:10]+'z.awp130pgrbf'+str(fhrs[j]).zfill(2)+'.grib2'
      #  grbs = pygrib.open(fil)
         grbs = grib2io.open(fil,mode='r')
      except:   
         fil = DATA_DIR+'/'+str.lower(model_str)+'.'+cycle[0:8]+'.t'+ \
               cycle[8:10]+'z.awp130pgrbf'+str(fhrs[j]).zfill(2)+'.grib2'
         grbs = pygrib.open(fil)

      HLwindow = 75
      skip = 10

      # Precip for RAP
#     precip = grbs.select(name='Total Precipitation')[0].values/25.4    # first precip message is run-total bucket; second is 1-h bucket


   #################
   # HRRR
   elif str.upper(model_str[0:4]) == 'HRRR':
      try:
         fil = '/gpfs/hps/nco/ops/com/hrrr/prod/hrrr.'+cycle[0:8]+'/conus/'+str.lower(model_str[0:6])+'.t'+ \
               cycle[8:10]+'z.wrfprsf'+str(fhrs[j]).zfill(2)+'.grib2'
     #   grbs = pygrib.open(fil)
         grbs = grib2io.open(fil,mode='r')
      except:   
         fil = DATA_DIR+'/'+str.lower(model_str)+'.'+cycle[0:8]+'/conus/'+ \
               str.lower(model_str)+'.t'+ \
               cycle[8:10]+'z.wrfprsf'+str(fhrs[j]).zfill(2)+'.grib2'
         grbs = pygrib.open(fil)

      HLwindow = 175
      skip = 25


   #  uh02 = grbs.select(parameterName='199',typeOfLevel='heightAboveGroundLayer',bottomLevel=0,topLevel=2000)[0].values

   #  vv02 = grbs.select(name='Vorticity (relative)',typeOfLevel='heightAboveGroundLayer',bottomLevel=0,topLevel=2000)[0].values
   #  vv01 = grbs.select(name='Vorticity (relative)',typeOfLevel='heightAboveGroundLayer',bottomLevel=0,topLevel=1000)[0].values


   # FV3-CAM
   elif str.upper(model_str) in FV3CAMs:
      try:
         fil = '/com2/nam/prod/nam.'+cycle[0:8]+'/'+str.lower(model_str[0:3])+'.t'+ \
               cycle[8:10]+'z.conusnest.hiresf'+str(fhrs[j]).zfill(2)+'.tm00.grib2'
         grbs = pygrib.open(fil)
      except:
         fil = DATA_DIR+'/'+str.upper(model_str)+'/'+str.lower(model_str[0:6])+'.'+cycle[0:8]+'.t'+ \
               cycle[8:10]+'z.conus.f'+str(fhrs[j]).zfill(2)+'.grib2'
         grbs = pygrib.open(fil)

      HLwindow = 300
      skip = 25


   # AQM
   elif str.upper(model_str) == 'AQM':
      try:
         fil = '/fake/site/ops/prod/com/gfs/'+os.environ['gfs_ver']+'/gfs.'+cycle[0:8]+'/'+cycle[8:10]+'/atmos/'+ \
               str.lower(model_str)+'.t'+cycle[8:10]+'z.pgrb2.0p25.f'+str(fhrs[j]).zfill(3)
         grbs = grib2io.open(fil,mode='r')
      except:
      #  fil = DATA_DIR+'/'+str.lower(model_str)+'.'+cycle[0:8]+'/'+cycle[8:10]+'/'+ \
      #        str.lower(model_str)+'.t'+cycle[8:10]+'z.cmaq.f'+str(fhrs[j]).zfill(3)+'.793.grib2'
      #  grbs = grib2io.open(fil,mode='r')
         fil = DATA_DIR+'/'+str.lower(model_str)+'.'+cycle[0:8]+'/'+cycle[8:10]+'/'+ \
               str.lower(model_str)+'.t'+cycle[8:10]+'z.met_sfc.f'+str(fhrs[j]).zfill(3)+'.nc'
         fil2 = DATA_DIR+'/'+str.lower(model_str)+'.'+cycle[0:8]+'/'+cycle[8:10]+'/'+ \
               str.lower(model_str)+'.t'+cycle[8:10]+'z.chem_sfc.f'+str(fhrs[j]).zfill(3)+'.nc'
         nc = netCDF4.Dataset(fil,'r')
         nc2 = netCDF4.Dataset(fil2,'r')
         lats = nc2.variables['lat'][:]
         lons = nc2.variables['lon'][:]
         hpbl = nc.variables['hpbl'][0,:,:]
         t2 = nc.variables['tmp2m'][0,:,:]*9/5 - 459.67
         u10 = nc.variables['ugrd10m'][0,:,:]*1.94384
         v10 = nc.variables['vgrd10m'][0,:,:]*1.94384
         nc.close()
         nc2.close()


  #   aotk = grbs.select(shortName='AOTK')[0].data



   #################

   '''
   if 't2' in fields:
       t2 = grbs.select(shortName='TMP',level='2 m above ground')[0].data*9/5 - 459.67

   if 'hpbl' in fields:
       hpbl = grbs.select(shortName='HPBL')[0].data

   if 'td2' in fields:
       td2 = grbs.select(shortName='DPT',level='2 m above ground')[0].data*9/5 - 459.67

   if '10mwind' in fields: 
       u10 = grbs.select(shortName='UGRD',level='10 m above ground')[0].data*1.94384
       v10 = grbs.select(shortName='VGRD',level='10 m above ground')[0].data*1.94384
       isotach_10m = np.sqrt(u10**2+v10**2)
   '''

#      if str.upper(model_str) == 'EC':
#          t2  = grbs.select(stepType='instant',name='2 metre temperature',level=0)[0].values*9/5 - 459.67
#          td2 = grbs.select(stepType='instant',name='2 metre dewpoint temperature',level=0)[0].values*9/5 - 459.67
#          u10 = grbs.select(stepType='instant',name='10 metre U wind component',level=0)[0].values*1.94384
#          v10 = grbs.select(stepType='instant',name='10 metre V wind component',level=0)[0].values*1.94384
#      else:
#          t2  = grbs.select(stepType='instant',name='2 metre temperature',typeOfLevel='heightAboveGround',level=2)[0].values*9/5 - 459.67
#          td2 = grbs.select(stepType='instant',name='2 metre dewpoint temperature',typeOfLevel='heightAboveGround',level=2)[0].values*9/5 - 459.67
#          u10 = grbs.select(stepType='instant',name='10 metre U wind component',typeOfLevel='heightAboveGround',level=10)[0].values*1.94384
#          v10 = grbs.select(stepType='instant',name='10 metre V wind component',typeOfLevel='heightAboveGround',level=10)[0].values*1.94384


#  if str.upper(model_str) != 'EC':
#     pw = grbs.select(name='Precipitable water')[0].values/25.4


   if str.upper(model_str) == 'EC':
      levels = [200,250,300,400,500,600,700,850]
   else:
      levels = np.arange(250,851,50)

#  for level in levels:
#     rh = grbs.select(name='Relative humidity',typeOfLevel='isobaricInhPa',level=level)[0].values
#     rh_avg.append(rh)

#  rh_avg = np.nanmean(np.array(rh_avg),axis=0)



#fields = ['wind500','vort500','500hght_slp_10mwind','10mwind','t2','td2']
   if 'wind500' in fields or 'vort500' in fields: 
       hghts_500 = grbs.select(shortName='HGT',level='500 mb')[0].data()*0.1
       if str.upper(model_str) != 'EC':
           vort_500 = grbs.select(shortName='ABSV',level='500 mb')[0].data()*1.e5
       uwind_500 = grbs.select(shortName='UGRD',level='500 mb')[0].data()*1.94384
       vwind_500 = grbs.select(shortName='VGRD',level='500 mb')[0].data()*1.94384
       isotach_500 = np.sqrt(uwind_500**2+vwind_500**2)

#  uwind_250 = grbs.select(name='U component of wind',typeOfLevel='isobaricInhPa',level=250)[0].values*1.94384
#  uwind_850 = grbs.select(name='U component of wind',typeOfLevel='isobaricInhPa',level=850)[0].values*1.94384
#  vwind_250 = grbs.select(name='V component of wind',typeOfLevel='isobaricInhPa',level=250)[0].values*1.94384
#  vwind_850 = grbs.select(name='V component of wind',typeOfLevel='isobaricInhPa',level=850)[0].values*1.94384

#  omega_700 = grbs.select(name='Vertical velocity',typeOfLevel='isobaricInhPa',level=700)[0].values*10.
#  omega_725 = grbs.select(name='Vertical velocity',typeOfLevel='isobaricInhPa',level=725)[0].values*10.
#  omega_750 = grbs.select(name='Vertical velocity',typeOfLevel='isobaricInhPa',level=750)[0].values*10.
#  omega_775 = grbs.select(name='Vertical velocity',typeOfLevel='isobaricInhPa',level=775)[0].values*10.
#  omega_800 = grbs.select(name='Vertical velocity',typeOfLevel='isobaricInhPa',level=800)[0].values*10.
#  omega_825 = grbs.select(name='Vertical velocity',typeOfLevel='isobaricInhPa',level=825)[0].values*10.
#  omega_850 = grbs.select(name='Vertical velocity',typeOfLevel='isobaricInhPa',level=850)[0].values*10.
#  omega_875 = grbs.select(name='Vertical velocity',typeOfLevel='isobaricInhPa',level=875)[0].values*10.
#  omega_900 = grbs.select(name='Vertical velocity',typeOfLevel='isobaricInhPa',level=900)[0].values*10.

#  hghts_700  = grbs.select(name='Geopotential Height',typeOfLevel='isobaricInhPa',level=700)[0].values*0.1
#  psfc = grbs.select(name='Surface pressure')[0].values*0.01

#  omega = np.nanmean(np.array(omega),axis=0)

#  soilw0_10 = grbs.select(name='Volumetric soil moisture content',scaledValueOfFirstFixedSurface=0)[0].values
#  land = grbs.select(name='Land-sea mask')[0].values


   if 'apcp' in fields or 'apcp24' in fields: 
       # Precip for NAM3 and HiResWNMMB
       if str.upper(model_str) == 'NAM3' or str.upper(model_str) == 'HIRESWNMMB':
          # first precip message is 3-h bucket; second is 1-h bucket
          if (fhrs[1]-fhrs[0] == 3): 
              precip_ind = 0
          elif (fhrs[1]-fhrs[0] == 1): 
              precip_ind = 1

          bucket = grbs.select(name='Total Precipitation')[precip_ind].values/25.4
          APCP.append(bucket)

       # Precip for HiResWARW/2
       elif str.upper(model_str[0:9]) == 'HIRESWARW' or str.upper(model_str) == 'RAP':
          gr = grbs.select(name='Total Precipitation')[0]
          print(gr)
          rt_bucket = grbs.select(name='Total Precipitation')[0].values/25.4
          if j == 0:
              APCP.append(rt_bucket)
          else:
            # bucket = rt_bucket - APCP[j-1]
              bucket = rt_bucket - np.sum(np.array(APCP),axis=0)
              APCP.append(bucket)

       # Precip for HREF means 
       if str.upper(model_str[0:4]) == 'HREF':
          # first precip message is 1-h bucket; second is 3-h bucket
          if (fhrs[1]-fhrs[0] == 3): 
              precip_ind = 1
          elif (fhrs[1]-fhrs[0] == 1): 
              precip_ind = 0

          grb = grbs.select(name='Total Precipitation')[precip_ind]
          bucket = grb.values/25.4
          APCP.append(bucket)


   if 'prob_ffri' in fields:

       prob_ffg1 = grbs.select(name='Probability of precipitation exceeding flash flood guidance values',upperLimit=1.0)[0].values
       fields = ['prob_ffg1']

       if fhrs[j]%3 == 0:
           try:
               prob_ffg3 = grbs.select(name='Probability of precipitation exceeding flash flood guidance values',upperLimit=3.0)[0].values
           except:
               prob_ffg3 = None
           if prob_ffg3 is not None:
               fields.extend(['prob_ffg3'])

       if fhrs[j] >= 6 and fhrs[j]%3 == 0:
           try:
               prob_ffg6 = grbs.select(name='Probability of precipitation exceeding flash flood guidance values',upperLimit=6.0)[0].values
           except:
               prob_ffg6 = None
           if prob_ffg6 is not None:
               fields.extend(['prob_ffg6'])

           try:
               prob_ari2_6h = grbs.select(name='Total Precipitation',lengthOfTimeRange=6,upperLimit=2.0)[0].values
           except:
               prob_ari2_6h = None
           if prob_ari2_6h is not None:
               fields.extend(['prob_ari2_6h'])

           try:
               prob_ari5_6h = grbs.select(name='Total Precipitation',lengthOfTimeRange=6,upperLimit=5.0)[0].values
           except:
               prob_ari5_6h = None
           if prob_ari5_6h is not None:
               fields.extend(['prob_ari5_6h'])

           try:
               prob_ari10_6h = grbs.select(name='Total Precipitation',lengthOfTimeRange=6,upperLimit=10.0)[0].values
           except:
               prob_ari10_6h = None
           if prob_ari10_6h is not None:
               fields.extend(['prob_ari10_6h'])

           try:
               prob_ari100_6h = grbs.select(name='Total Precipitation',lengthOfTimeRange=6,upperLimit=100.0)[0].values
           except:
               prob_ari100_6h = None
           if prob_ari100_6h is not None:
               fields.extend(['prob_ari100_6h'])


       if fhrs[j] >= 24 and fhrs[j]%3 == 0:
           try:
               prob_ari2_24h   = grbs.select(name='Total Precipitation',lengthOfTimeRange=24,upperLimit=2.0)[0].values
           except:
               prob_ari2_24h = None
           if prob_ari2_24h is not None:
               fields.extend(['prob_ari2_24h'])

           try:
               prob_ari5_24h   = grbs.select(name='Total Precipitation',lengthOfTimeRange=24,upperLimit=5.0)[0].values
           except:
               prob_ari5_24h = None
           if prob_ari5_24h is not None:
               fields.extend(['prob_ari5_24h'])

           try:
               prob_ari10_24h   = grbs.select(name='Total Precipitation',lengthOfTimeRange=24,upperLimit=10.0)[0].values
           except:
               prob_ari10_24h = None
           if prob_ari10_24h is not None:
               fields.extend(['prob_ari10_24h'])

           try:
               prob_ari100_24h   = grbs.select(name='Total Precipitation',lengthOfTimeRange=24,upperLimit=100.0)[0].values
           except:
               prob_ari100_24h = None
           if prob_ari100_24h is not None:
               fields.extend(['prob_ari100_24h'])




   '''
   grbs.seek(0)
   msgs = grbs['CAPE']
   print(msgs[0])

   for gr in grbs:
       if gr.shortName == 'CAPE':
           print(gr)
           exit()
   '''

   if 'refc' in fields or 'refc_uh' in fields: 
       if str.upper(model_str[0:4]) != 'HREF': 
       #   refc = grbs.select(name='Maximum/Composite radar reflectivity')[0].values
           refc = grbs.select(shortName='REFC')[0].data()

       if str.upper(model_str[0:4]) in HiResModels and fhrs[j] > 0:
           uh25 = grbs.select(parameterName='199',typeOfLevel='heightAboveGroundLayer',bottomLevel=2000,topLevel=5000)[0].values
           uh25 = grbs.select(shortName='MXUPHL')
           if fhrs[j] == 0:
               uh25 = uh25*0.
      #    UH25_accum.append(uh25)


   if 'uh25' in fields or 'UH25_accum' in accums: 
      uh25 = grbs.select(parameterName='199',typeOfLevel='heightAboveGroundLayer',bottomLevel=2000,topLevel=5000)[0].values
      if fhrs[j] == 0:
          uh25 = uh25*0.
      if 'UH25_accum' in accums: 
          UH25_accum.append(uh25)

   if 'uh03' in fields or 'UH03_accum' in accums: 
      uh03 = grbs.select(parameterName='199',typeOfLevel='heightAboveGroundLayer',bottomLevel=0,topLevel=3000)[0].values
      if fhrs[j] == 0:
         uh03 = uh03*0.
      if 'UH03_accum' in accums: 
          UH03_accum.append(uh03)

   if 'uh02' in fields or 'UH02_accum' in accums and str.upper(model_str[0:4]) == 'HRRR': 
      uh02 = grbs.select(parameterName='199',typeOfLevel='heightAboveGroundLayer',bottomLevel=0,topLevel=2000)[0].values
      if fhrs[j] == 0:
         uh02 = uh02*0.
      if 'UH02_accum' in accums: 
          UH02_accum.append(uh02)

   if 'vv02' in fields or 'VV02_accum' in accums: 
      vv02 = grbs.select(name='Vorticity (relative)',typeOfLevel='heightAboveGroundLayer',bottomLevel=0,topLevel=2000)[0].values
      maxref = grbs.select(parameterName='198',typeOfLevel='heightAboveGround',level=1000)[0].values
      if fhrs[j] == 0:
         vv02 = vv02*0.
    # vv02[maxref < 35] = 0.
      if 'VV02_accum' in accums: 
          VV02_accum.append(vv02)

   if 'vv01' in fields or 'VV01_accum' in accums: 
      vv01 = grbs.select(name='Vorticity (relative)',typeOfLevel='heightAboveGroundLayer',bottomLevel=0,topLevel=1000)[0].values
      if fhrs[j] == 0:
         vv01 = vv01*0.
    # vv01[maxref < 35] = 0.
      if 'VV01_accum' in accums: 
          VV01_accum.append(vv01)

#  if str.upper(model_str) == 'HRRR' or str.upper(model_str[0:6]) == 'HIRESW':
#     uh03 = grbs.select(parameterName='199',typeOfLevel='heightAboveGroundLayer',bottomLevel=0,topLevel=3000)[0].values

   if 'sbcape_sbcin' in fields: 
       sbcape = grbs.select(shortName='CAPE',level='surface')[0].data()
       sbcin = grbs.select(shortName='CIN',level='surface')[0].data()

   if 'mlcape_mlcin' in fields: 
       mlcape = grbs.select(shortName='CAPE',level='90-0 mb above ground')[0].data()
       mlcin  = grbs.select(shortName='CIN',level='90-0 mb above ground')[0].data()

   if 'slp_mucape_shear06' in fields: 
       if str.upper(model_str[0:6]) == 'HIRESW': 
           mucape = grbs.select(shortName='CAPE',level='180-0 mb above ground')[0].data()
       else:
       #   mucape = grbs.select(name='Convective available potential energy',typeOfLevel='pressureFromGroundLayer',topLevel=25500)[0].values
           mucape = grbs.select(shortName='CAPE',level='255-0 mb above ground')[0].data()
           mucin  = grbs.select(shortName='CIN',level='255-0 mb above ground')[0].data()
       #   mucin = grbs.select(name='Convective inhibition',typeOfLevel='pressureFromGroundLayer',topLevel=25500)[0].values

       if str.upper(model_str) != 'GFS' and str.upper(model_str) != 'NAM' and str.upper(model_str[4:]) != 'AVRG':
           u_shr06 = grbs.select(shortName='VUCSH')[0].data()*1.94384
           v_shr06 = grbs.select(shortName='VVCSH')[0].data()*1.94384

   if 'lr75' in fields:
       temp_500 = grbs.select(name='Temperature',typeOfLevel='isobaricInhPa',level=500)[0].values-273.15
       temp_700 = grbs.select(name='Temperature',typeOfLevel='isobaricInhPa',level=700)[0].values-273.15
       hghts_500  = grbs.select(name='Geopotential Height',typeOfLevel='isobaricInhPa',level=500)[0].values*0.1
       hghts_700  = grbs.select(name='Geopotential Height',typeOfLevel='isobaricInhPa',level=700)[0].values*0.1
       lr75 = (temp_700 - temp_500)*1000./(hghts_500 - hghts_700)/9.81


   '''
   if str.upper(model_str[0:3]) == 'RAP' or str.upper(model_str[0:4]) == 'HRRR':
      grb = grbs.select(shortName='MSLMA')[0]
   elif str.upper(model_str[0:2]) == 'EC':
      grb = grbs.select(name='Mean sea level pressure')[0]
   elif str.upper(model_str) == 'AQM':
#     grb = grbs.select(shortName='AOTK')[0]
      grb = grbs.select(shortName='TMP',level='2 m above ground')[0]
   elif str.upper(model_str[0:4]) != 'HREF':
      try:
         grb = grbs.select(shortName='MSLET')[0]
      except:
         grb = grbs.select(shortName='PRMSL')[0]
   if str.upper(model_str[0:4]) != 'HREF' or str.upper(model_str) != 'AQM':
   #  mslp = grb.values*0.01
      mslp = grb.data*0.01
      print(grb.fullName)


   if str.upper(model_str[0:4]) == 'HREF' and fhrs[j] == 0:
      sys.exc_clear()
   else:
      lats, lons = grb.latlons()

   grbs.close()
   '''

   if model_str[0:2] == 'EC' and 'vort500' in fields:
      f = 2*7.292E-5*np.sin(lats*(np.pi/180.))
      vort_500 = (relv_500 + f)*1.e5


   main()


exit()

UH25_accum = np.amax(np.array(UH25_accum),axis=0)
UH03_accum = np.amax(np.array(UH03_accum),axis=0)
VV02_accum = np.amax(np.array(VV02_accum),axis=0)
VV01_accum = np.amax(np.array(VV01_accum),axis=0)

plots = [n for n in itertools.product(domains,accums)]
plot_fields(('DC-NYC','UH25_accum'))

main()

