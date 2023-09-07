#!/usr/bin/env python

# Run as:
# python plot_TC_sameinit.py $CYCLE $TC_name/ID/Year
# python plot_TC_sameinit.py 2018090100 FlorenceAL062018
#


import numpy as np
import os, sys, datetime, time, subprocess
import re, csv, glob
import multiprocessing, itertools, collections
import scipy, ncepy

import pyproj
import cartopy.crs as ccrs
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

import matplotlib
import matplotlib.image as image
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib import colors as c
matplotlib.use('Agg')

import cartopy
import dawsonpy


# Function to interpolate over nans in 
def interpolate_gaps(values, limit=None):
    """
    Fill gaps using linear interpolation, optionally only fill gaps up to a
    size of `limit`.
    """
    values = np.asarray(values)
    i = np.arange(values.size)
    valid = np.isfinite(values)
    filled = np.interp(i, i[valid], values[valid])

    if limit is not None:
        invalid = ~valid
        for n in range(1, limit+1):
            invalid[:-n] &= invalid[n:]
        filled[invalid] = np.nan

    return filled


# Function to smooth curves
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



reader = shpreader.Reader('/apps/prod/python-modules/3.8.6/intel/19.1.3.304/lib/python3.8/site-packages/cartopy/data/shapefiles/USGS/shp/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())


# Determine cycle initialization date/time
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

cycle_date = datetime.datetime(YYYY,MM,DD,HH,0,0)



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
print(TC_name, TC_number)



# Option to make special HAFS comparison graphics
try:
   dummy = str(sys.argv[3])
   do_hafs = True
   print('Plotting HAFS comparison images')
except IndexError:
   dummy = None
   do_hafs = False



# Set path and create graphx directory (if not already created)

BDECK_DIR = '/lfs/h1/ops/prod/dcom/nhc/atcf-noaa/btk'
DATA_DIR = os.path.join('/lfs/h2/emc/vpppg/noscrub/',os.environ['USER'],'MEG', TC_name, 'data')

GRAPHX_DIR = os.path.join('/lfs/h2/emc/ptmp',os.environ['USER'],'MEG', TC_name, 'graphx')
LOWS_DIR = os.path.join('/lfs/h2/emc/ptmp',os.environ['USER'],'MEG', TC_name, 'graphx',cycle)

if os.path.exists(DATA_DIR):
   if not os.path.exists(GRAPHX_DIR):
      os.makedirs(GRAPHX_DIR)
else:
   raise NameError('data for '+TC_name+' not found')


#OUT_DIR = os.path.join(GRAPHX_DIR, cycle)
if do_hafs:
    OUT_DIR = GRAPHX_DIR + '/hafs'
else:
    OUT_DIR = GRAPHX_DIR
if not os.path.exists(OUT_DIR):
      os.makedirs(OUT_DIR)


# Set Landfall Date based on NHC 
# Needed to only look for RI periods prior to this time
#landfall_date = datetime.datetime(2020,8,25,0,0,0)
#landfall_date2 = datetime.datetime(2020,11,6,12,0,0)
#landfall_date = datetime.datetime(2021,8,29,16,55,0)
#landfall_date2 = datetime.datetime(2020,8,29,16,55,0)
landfall_date = datetime.datetime(2022,9,28,18,0,0)
landfall_date2 = datetime.datetime(2022,9,28,18,0,0)


try:
   valid_time
except NameError:
   valid_time = None



# Get list of cycles based on matching files in DATA_DIR
filelist = [f for f in glob.glob(DATA_DIR+'/'+str.lower(TC_name)+'_*_'+cycle+'.csv')]



mlats=[]
mlons=[]
mpres=[]
mvmax=[]
#mrmw=[]
mfhrs=[]
mtimes=[]
mnames=[]
line_colors=[]
models=['GFS','FV3GFS','EC','UK','CMC','HWRF','HMON']
models=['GFS','GFSO','EC','UK','CMC','HWRF','HMON']
#models=['GFS','GFSO','EC','UK','CMC']
#potential_colors=['red','blue','green','cyan','limegreen']

if do_hafs:
    models=['HWRF','HMON','HF3A','HF3S']
    potential_colors=['#8400C8','#00DC00','magenta','cyan']
else:
    models=['GEFSMEAN','GFS','ECMEAN','ECMO','CMC','UK','HWRF','HMON']
    potential_colors=['black','black','#FB2020','#FB2020','#1E3CFF','#E69F00','#8400C8','#00DC00']



i = 0
for model_str in models:

   if model_str == 'EC':
      try:
         print("trying to read ECMO data")
         track_file = DATA_DIR+'/'+str.lower(TC_name)+'_'+str.lower(model_str)+'_'+cycle+'.csv'
         f = open(track_file,'r')
         print("reading ECMO data")
      except:
         print("ECMO data not found. Trying EMX")
         try:
            track_file = DATA_DIR+'/'+str.lower(TC_name)+'_emx_'+cycle+'.csv'
            f = open(track_file,'r')
            print("reading EMX data")
            model_str = 'EMX'
         except:
            print("EMX data not found. Trying EC_DET")
            try:
               track_file = DATA_DIR+'/'+str.lower(TC_name)+'_ec_det_'+cycle+'.csv'
               f = open(track_file,'r')
               print("reading EC_DET data")
               model_str = 'EC_DET'
            except:
                print("EC_DET data not found. No EC data to plot")


   track_file = DATA_DIR+'/'+str.lower(TC_name)+'_'+str.lower(model_str)+'_'+cycle+'.csv'

   # lat/lon lists for each model
   clat=[]
   clon=[]
   cpres=[]
   cvmax=[]
#  crmw=[]
   cfhr=[]
   ctime=[]

   if os.path.exists(track_file):
      print(track_file)
      with open(track_file,'r') as f:
         reader=csv.reader(f)
         for row in reader:
            fcst_time = cycle_date+datetime.timedelta(hours=int(row[0]))
            if valid_time is not None:
               if fcst_time <= valid_time:
                  cfhr.append(float(row[0]))
                  clat.append(float(row[2]))
                  if float(row[3]) > 0:
                     clon.append(float(row[3])-360)
                  else:
                     clon.append(float(row[3]))
                  cpres.append(float(row[4]))
                  cvmax.append(float(row[5]))
       #          crmw.append(float(row[6]))
                  ctime.append(fcst_time)
            else:
               cfhr.append(float(row[0]))
               clat.append(float(row[2]))
               if float(row[3]) > 0:
               #  clon.append(float(row[3])-360)
                  clon.append(float(row[3]))
               else:
                  clon.append(float(row[3]))
               cpres.append(float(row[4]))
               cvmax.append(float(row[5]))
       #       crmw.append(float(row[6]))
               ctime.append(fcst_time)


    # if model_str == 'UKMet':
    #    print(len(clat))
    #    print(len(clon))
    #    print(len(cpres))
    #    print(len(cvmax))
    #    print(ctime)

      mlats.append(clat)
      mlons.append(clon)
      mpres.append(cpres)
      mvmax.append(cvmax)
#     mrmw.append(crmw)
      mfhrs.append(cfhr)
      mtimes.append(ctime)
      line_colors.append(potential_colors[i])
      if str.lower(model_str) == 'ec_det' or str.lower(model_str) == 'emx':
         mnames.append('EC') 
      else:
         mnames.append(model_str)

   i += 1

print(len(mlats))
#print(len(mlons[3]))
#print(len(mpres[3]))
#print(len(mvmax[3]))



final_valid_date = cycle_date
for i in range(len(mlats)):
   try:
      if final_valid_date < mtimes[i][-1]:
         final_valid_date = mtimes[i][-1]
   except:
      print('no times for '+models[i])



# Get observed data from Best Track file
# Get final valid time by picking last time of TD strength in Best Track file
olat = []
olon = []
opres = []
ovmax = []
otime = []

with open(BDECK_DIR+'/b'+str.lower(TC_number)+str(YYYY)+'.dat','r') as f:
   reader = csv.reader(f)
   for row in reader:
    # if row[10].replace(" ","")!='FAKE STRING':
    # if row[10].replace(" ","")=='TD' or row[10].replace(" ","")=='TS' or row[10].replace(" ","")=='HU':
    # if row[10].replace(" ","")=='TD' or row[10].replace(" ","")=='TS' or row[10].replace(" ","")=='HU' or row[10].replace(" ","")=='EX':
      if row[10].replace(" ","")=='DB' or row[10].replace(" ","")=='LO' or row[10].replace(" ","")=='TD' or row[10].replace(" ","")=='TS' or row[10].replace(" ","")=='HU' or row[10].replace(" ","")=='EX':
         if row[11].replace(" ","")=='34' or row[11].replace(" ","")=='0':
   #        print(row)
            rowtime=datetime.datetime.strptime(row[2].replace(" ",""),"%Y%m%d%H")

            if rowtime >= cycle_date and rowtime <= final_valid_date:
               olat.append(float(re.sub("N","",row[6]))/10.0)
               try:
                  olon.append(float(re.sub("W","",row[7]))/-10.0)
               except:
                  olon.append(float(re.sub("E","",row[7]))/10.0)
               ovmax.append(float(row[8]))
               opres.append(float(row[9]))
               otime.append(rowtime)



# Option to set final valid time to end of Best Track
match_end_bt = True
if match_end_bt:
   final_valid_date = otime[-1]

if cycle == '2019083000':
    final_valid_date = otime[24]
elif cycle == '2019083012':
    final_valid_date = otime[22]
elif cycle == '2019083100':
    final_valid_date = otime[20]
print('Final Valid date is : ',final_valid_date)

# Get list of dates for pressure/wind traces
init_inc = 6
date_list = []
temp_cycle_date = cycle_date

while temp_cycle_date <= final_valid_date:
   date_list.append(temp_cycle_date)
   temp_cycle_date += datetime.timedelta(hours=init_inc)

print(date_list[0], date_list[-1])

plot_opres=[]
plot_ovmax=[]
plot_ospeed=[]
plot_olf=[]
k = 1

for j in range(len(date_list)):
   for i in range(len(opres)):
      if otime[i] == date_list[j]:
         plot_opres.append(opres[i])
         plot_ovmax.append(ovmax[i])

         if otime[i] == landfall_date or otime[i] == landfall_date2:
            plot_olf.append(1)
         else:
            plot_olf.append(0)

         if i == 0:
            speed = np.nan
         else:
            speed = ncepy.gc_dist(olat[i-1],olon[i-1],olat[i],olon[i])/21600.*1.94384
         plot_ospeed.append(speed)


   if len(plot_opres) < k:
      plot_opres.append(np.nan)
      plot_ovmax.append(np.nan)
      plot_ospeed.append(np.nan)
      plot_olf.append(np.nan)

   k += 1

try:
   plot_olf_ind = [i for i, x in enumerate(plot_olf) if x == 1]
   markers_on = [plot_olf_ind]
except:
   markers_on = []

print(plot_olf_ind)

plot_mlats=[]
plot_mlons=[]
plot_mfhrs=[]
plot_mpres=[]
plot_mvmax=[]
plot_mspeed=[]


for x in range(len(mlats)):
   k = 1
   temp_mpres=[]
   temp_mvmax=[]
   temp_mspeed=[]
   temp_mlats=[]
   temp_mlons=[]
   temp_mfhrs=[]
   for j in range(len(date_list)):
      for i in range(len(mpres[x])):
         if mtimes[x][i] == date_list[j]:
            temp_mpres.append(mpres[x][i])
            temp_mvmax.append(mvmax[x][i])
            temp_mlats.append(mlats[x][i])
            temp_mlons.append(mlons[x][i])
            temp_mfhrs.append(mfhrs[x][i])

            if i == 0:
               speed = np.nan
            elif str.upper(mnames[x]) == 'HMON':
               speed = ncepy.gc_dist(mlats[x][i-2],mlons[x][i-2],mlats[x][i],mlons[x][i])/21600.*1.94384
            elif str.upper(mnames[x][0:2]) == 'UK':
               speed = ncepy.gc_dist(mlats[x][i-1],mlons[x][i-1],mlats[x][i],mlons[x][i])/43200.*1.94384
            else:
               speed = ncepy.gc_dist(mlats[x][i-1],mlons[x][i-1],mlats[x][i],mlons[x][i])/21600.*1.94384
            temp_mspeed.append(speed)


      if len(temp_mpres) < k:
         temp_mpres.append(np.nan)
         temp_mvmax.append(np.nan)
         temp_mspeed.append(np.nan)
       # temp_mlats.append(np.nan)
       # temp_mlons.append(np.nan)

      k += 1

   plot_mpres.append(temp_mpres)
   plot_mvmax.append(temp_mvmax)
   plot_mspeed.append(temp_mspeed)
   plot_mlats.append(temp_mlats)
   plot_mlons.append(temp_mlons)
   plot_mfhrs.append(temp_mfhrs)








def plot_tracks(domain):

   print('plotting '+cycle+' '+TC_name+' on '+domain+' domain')

   fig = plt.figure(figsize=(10.9,8.9))
   gs = GridSpec(1,1,wspace=0.0,hspace=0.0)

   # Define where Cartopy maps are located
   cartopy.config['data_dir'] = '/lfs/h2/emc/vpppg/save/'+os.environ['USER']+'/python/NaturalEarth'

   back_res='50m'
   back_img='off'


   if str.upper(domain) == 'CONUS':
       llcrnrlon = -121.5 
       llcrnrlat = 22.
       urcrnrlon = -64.5
       urcrnrlat = 48.
       proj = 'lcc'
       lon0 =-101.
       lat0 = 39.
       standard_parallels=(32,46)
       draw_counties = False
   elif str.upper(domain) == 'GOM':
       llcrnrlon = -100. 
       llcrnrlat = 12.5
       urcrnrlon = -70.
       urcrnrlat = 35.
       proj = 'lcc'
       lon0 =-87.5
       lat0 = 35.5
       standard_parallels=(25,46)
       draw_counties = False
   elif str.upper(domain) == 'MICHAEL':
       llcrnrlon = -90.
       llcrnrlat = 20.    # 17.5 or 25.
       urcrnrlon = -60.   # -65.
       urcrnrlat = 40.    # 45.
       proj = 'lcc'
       draw_counties = False
   elif str.upper(domain) == 'SE_COAST':
       llcrnrlon = -90.
       llcrnrlat = 22.    # 24.
       urcrnrlon = -72.   # -62.5, -67.5
       urcrnrlat = 37.5   # 42
       proj = 'lcc'
       lon0 =-81.
       lat0 = 39.
       standard_parallels=(32,46)
       draw_counties = False
   elif str.upper(domain) == 'IDA':
       llcrnrlon = -100. 
       llcrnrlat = 10.
       urcrnrlon = -60.
       urcrnrlat = 45.
       proj = 'merc'
       lat_ts = 35.
       draw_counties = False
   elif str.upper(domain) == 'IAN':
       llcrnrlon = -100. 
       llcrnrlat = 5.
       urcrnrlon = -52.
       urcrnrlat = 45.
       proj = 'merc'
       lon0 = -76.
       lat_ts = 35.
       draw_counties = False
   elif str.upper(domain) == 'CARIB':
       llcrnrlon = -80.
       llcrnrlat = 10.
       urcrnrlon = -55.
       urcrnrlat = 27.5
       proj = 'merc'
       lat_ts = 20.
       draw_counties = False
   elif str.upper(domain) == 'GR_ANT':
       llcrnrlon = -85.
       llcrnrlat = 15.
       urcrnrlon = -62.5
       urcrnrlat = 32.
       proj = 'merc'
       lon0 = -73.75
       lat_ts = 20.
       draw_counties = False
   elif str.upper(domain) == 'BAHAMAS':
       llcrnrlon = -85.
       llcrnrlat = 22.
       urcrnrlon = -72.5
       urcrnrlat = 30.
       proj = 'merc'
       lat_ts = 25.
       draw_counties = False
   elif str.upper(domain) == 'CPAC':
       llcrnrlon = -180.
       llcrnrlat = 5.
       urcrnrlon = -110.
       urcrnrlat = 40.
       proj = 'merc'
       lat_ts = 20.
       draw_counties = False
   elif str.upper(domain) == 'HAWAII':
       llcrnrlon = -165.
       llcrnrlat = 10.
       urcrnrlon = -145.
       urcrnrlat = 25.
       proj = 'merc'
       lon0 = -155.
       lat_ts = 20.
       draw_counties = False


   extent = [llcrnrlon-1,urcrnrlon+1,llcrnrlat-1,urcrnrlat+1]

   if proj == 'lcc':
       myproj = ccrs.LambertConformal(central_longitude=lon0,central_latitude=lat0,
                false_easting=0.0,false_northing=0.0,secant_latitudes=None,
                standard_parallels=standard_parallels,globe=None)
   elif proj == 'merc':
       myproj = ccrs.Mercator(central_longitude=lon0,min_latitude=llcrnrlat,
                max_latitude=urcrnrlat,latitude_true_scale=lat_ts,globe=None)

   # All lat lons are earth relative, so setup the associated projection correct for that data
   transform = ccrs.PlateCarree()

   ax = fig.add_subplot(gs[0:1,0:1], projection=myproj)
   ax.set_extent(extent,crs=transform)
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
   ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidths=0.5,linestyle='solid',edgecolor='k',zorder=4)
   ax.add_feature(cfeature.STATES.with_scale('50m'),linewidths=0.3,linestyle='solid',edgecolor='k',zorder=4)

   latlongrid = 10.
   if draw_counties:
      ax.add_feature(COUNTIES,facecolor='none',edgecolor='gray')
      latlongrid = 5.
   if str.upper(domain) == 'BARRY':
      latlongrid = 5.

   parallels = np.arange(0.,90.,latlongrid)
   meridians = np.arange(180.,360.,latlongrid)
   print(parallels)


   if str.upper(domain) == 'IAN':

    #  xgl = ax.gridlines(crs=transform,draw_labels=True,xlocs=meridians,
    #                     linewidth=0.3,color='black',linestyle='solid')
       gl = ax.gridlines(crs=transform,draw_labels=True,
                         linewidth=0.3,color='black',linestyle='solid')
       gl.top_labels = False
       gl.right_labels = False
       gl.xlocater = mticker.FixedLocator(meridians)
       gl.ylocater = mticker.FixedLocator(parallels)
     # gl.xformatter = LONGITUDE_FORMATTER
     # gl.yformatter = LATITUDE_FORMATTER
       gl.xformatter = LongitudeFormatter()
       gl.yformatter = LatitudeFormatter()
       gl.xlabel_style = {'size':10, 'color':'black'}
       gl.ylabel_style = {'size':10, 'color':'black'}


#  print(np.shape(olat))

  #for i in range(len(mlats)):
   for i in range(len(plot_mlats)):
      label_str = str.upper(mnames[i])
      if str.upper(mnames[i]) == 'GFS':
         label_str = 'GFS'
      elif str.upper(mnames[i]) == 'GFSO':
         label_str = 'GFSv14'
      elif str.upper(mnames[i]) == 'GEFSMEAN':
         label_str = 'GEFS Mean'
      elif str.upper(mnames[i]) == 'ECMEAN':
         label_str = 'EC Mean'
      elif str.upper(mnames[i]) == 'EMX' or str.upper(mnames[i]) == 'ECMO':
         label_str = 'EC'
      elif str.upper(mnames[i]) == 'UKMET' or str.upper(mnames[i]) == 'UKX' or str.upper(mnames[i]) == 'UK':
         label_str = 'UKM'
      elif str.upper(mnames[i]) == 'HF3A':
         label_str = 'HAFS v0.3A'
      elif str.upper(mnames[i]) == 'HF3S':
         label_str = 'HAFS v0.3S'

      if str.upper(mnames[i][-4:]) == 'MEAN':
         plt.plot(plot_mlons[i], plot_mlats[i], '--', color=line_colors[i], label=label_str, linewidth=2.,transform=transform)
      else:
         plt.plot(plot_mlons[i], plot_mlats[i], '-', color=line_colors[i], label=label_str, linewidth=2.,transform=transform)

   plt.plot(olon, olat, '-', color='#696969', label='BEST', linewidth=3.,transform=transform)

   if str.upper(domain) == 'SE_COAST':
      plt.legend(loc="upper left",framealpha=1.0)
   elif str.upper(domain) == 'NW_ATL':
      plt.legend(loc="upper left",framealpha=1.0)
   elif str.upper(domain) == 'NATL':
      plt.legend(loc="upper right",framealpha=1.0)
   elif str.upper(domain) == 'ATL':
      plt.legend(loc="upper right",framealpha=1.0)
   elif str.upper(domain) == 'CPAC':
      plt.legend(loc="upper right",framealpha=1.0)
   elif str.upper(domain) == 'HAWAII':
      plt.legend(loc="upper right",framealpha=1.0)
   elif str.upper(domain) == 'MICHAEL':
      plt.legend(loc="lower right",framealpha=1.0)
   elif str.upper(domain) == 'LOUISIANA':
      plt.legend(loc="center right",framealpha=1.0)
   elif str.upper(domain) == 'BARRY':
      plt.legend(loc="center right",framealpha=1.0)
   elif str.upper(domain) == 'GOM':
      plt.legend(loc="upper right",framealpha=1.0)
   elif str.upper(domain) == 'IDA':
      plt.legend(loc="center right",framealpha=1.0)
   elif str.upper(domain) == 'IAN':
      plt.legend(loc="center right",framealpha=1.0)
   elif str.upper(domain) == 'ECONUS':
      plt.legend(loc="lower left",framealpha=1.0)


   titlestr1 = cycle_date.strftime('Hurricane '+TC_name+' Tracks - %HZ %d %B %Y Initializations')
   titlestr2 = final_valid_date.strftime('Valid through %HZ %d %B %Y')
   plt.text(0.5, 1.05, titlestr1, fontweight='bold', horizontalalignment='center', transform=ax.transAxes)
   plt.text(0.5, 1.01, titlestr2, fontweight='bold', horizontalalignment='center', transform=ax.transAxes)

   fname = 'bad'+str.lower(TC_name)+'_tracks_'+str.lower(domain)+'_'+cycle

   plt.savefig(OUT_DIR+'/'+fname+'.png',bbox_inches='tight')
   plt.close()






def ptrace():

   print('plotting pressure traces')
   print(len(plot_opres))    # same length as date_list

   fig = plt.figure(figsize=(9,8))

   for i in range(len(plot_mpres)):
      label_str = str.upper(mnames[i])
      if str.upper(mnames[i]) == 'GFS':
         label_str = 'GFS'
      elif str.upper(mnames[i]) == 'GFSO':
         label_str = 'GFSv14'
      elif str.upper(mnames[i]) == 'GEFSMEAN':
         label_str = 'GEFS Mean'
      elif str.upper(mnames[i]) == 'ECMEAN':
         label_str = 'EC Mean'
      elif str.upper(mnames[i]) == 'EMX' or str.upper(mnames[i]) == 'ECMO':
         label_str = 'EC'
      elif str.upper(mnames[i]) == 'UKMET' or str.upper(mnames[i]) == 'UKX' or str.upper(mnames[i]) == 'UK':
         label_str = 'UKM'
      elif str.upper(mnames[i]) == 'HF3A':
         label_str = 'HAFS v0.3A'
      elif str.upper(mnames[i]) == 'HF3S':
         label_str = 'HAFS v0.3S'

      if str.upper(mnames[i][-4:]) == 'MEAN':
         plt.plot(plot_mpres[i], '--', color=line_colors[i], label=label_str, linewidth=2.)
      elif str.upper(mnames[i][0:2]) == 'UK':
         try:
             filled = interpolate_gaps(plot_mpres[i], limit=2)
             plt.plot(filled, '-', color=line_colors[i], label=label_str, linewidth=2.)
         except:
             plt.plot(plot_mpres[i], '-', color=line_colors[i], label=label_str, linewidth=2.)
      else:
         plt.plot(plot_mpres[i], '-', color=line_colors[i], label=label_str, linewidth=2.)


   plt.plot(plot_opres, '-', color='#696969', label='BEST', linewidth=3., marker='D', markersize=7, markevery=markers_on)

   xlen = len(plot_opres) 
   x = np.arange(0,xlen,1)

   plt.axis([0,xlen,920,1020])
   plt.axhspan(900, 1020, facecolor='0.5', alpha=0.5)

   labels=[]
   for x in range(0,xlen,2):
      if x%4 == 0:
         labels.append(date_list[x].strftime('%HZ %m/%d'))
      else:
         labels.append('')
 
#  labels = [date_list[x].strftime('%HZ %m/%d') for x in range(0,xlen,6)]
   plt.xticks(np.arange(0,xlen,step=2),labels,rotation=-45,ha='left')
   plt.ylabel('Minimum Pressure (mb)')

   plt.grid(True)

   if int(cycle[0:8]) <= 20220924: 
       plt.legend(loc="lower left")
   elif int(cycle[0:8]) >= 20220925: 
       plt.legend(loc="lower right")

   titlestr = 'Hurricane '+TC_name+' Minimum Pressure Traces \n'+ \
              cycle_date.strftime('%HZ %d %b Initializations')+' valid through '+final_valid_date.strftime('%HZ %d %b %Y')
   plt.title(titlestr, fontweight='bold')

   fname = str.lower(TC_name)+'_ptrace_'+cycle

   plt.savefig(OUT_DIR+'/'+fname+'.png',bbox_inches='tight')
   plt.close()




def vtrace():

   print('plotting wind traces')
   print(len(plot_ovmax))    # same length as date_list

   fig = plt.figure(figsize=(9,8))

   for i in range(len(plot_mvmax)):
      label_str = str.upper(mnames[i])
      if str.upper(mnames[i]) == 'GFS':
         label_str = 'GFS'
      elif str.upper(mnames[i]) == 'GFSO':
         label_str = 'GFSv14'
      elif str.upper(mnames[i]) == 'GEFSMEAN':
         label_str = 'GEFS Mean'
      elif str.upper(mnames[i]) == 'ECMEAN':
         label_str = 'EC Mean'
      elif str.upper(mnames[i]) == 'EMX' or str.upper(mnames[i]) == 'ECMO':
         label_str = 'EC'
      elif str.upper(mnames[i]) == 'UKMET' or str.upper(mnames[i]) == 'UKX' or str.upper(mnames[i]) == 'UK':
         label_str = 'UKM'
      elif str.upper(mnames[i]) == 'HF3A':
         label_str = 'HAFS v0.3A'
      elif str.upper(mnames[i]) == 'HF3S':
         label_str = 'HAFS v0.3S'

      if str.upper(mnames[i][-4:]) == 'MEAN':
         plt.plot(plot_mvmax[i], '--', color=line_colors[i], label=label_str, linewidth=2.)
      elif str.upper(mnames[i][0:2]) == 'UK':
         try:
             filled = interpolate_gaps(plot_mvmax[i], limit=2)
             plt.plot(filled, '-', color=line_colors[i], label=label_str, linewidth=2.)
         except:
             plt.plot(plot_mvmax[i], '-', color=line_colors[i], label=label_str, linewidth=2.)
      else:
         plt.plot(plot_mvmax[i], '-', color=line_colors[i], label=label_str, linewidth=2.)

   plt.plot(plot_ovmax, '-', color='#696969', label='BEST', linewidth=3., marker='D', markersize=7, markevery=markers_on)


   xlen = len(plot_ovmax) 
   x = np.arange(0,xlen,1)

   plt.axis([0,xlen,5,160])
   plt.axhspan(0, 34, facecolor='0.5', alpha=0.5)
   plt.axhspan(34, 64, facecolor='0.4', alpha=0.5)
   plt.axhspan(64, 83, facecolor='0.3', alpha=0.5)
   plt.axhspan(83, 96, facecolor='0.25', alpha=0.5)
   plt.axhspan(96, 113, facecolor='0.2', alpha=0.5)
   plt.axhspan(113, 137, facecolor='0.15', alpha=0.5)
   plt.axhspan(137, 200, facecolor='0.1', alpha=0.5)

   labels = []
   for x in range(0,xlen,2):
      if x%4 == 0:
         labels.append(date_list[x].strftime('%HZ %m/%d'))
      else:
         labels.append('')

#  labels = [date_list[x].strftime('%HZ %m/%d') for x in range(0,xlen,6)]
   plt.xticks(np.arange(0,xlen,step=2),labels,rotation=-45,ha='left')
   plt.ylabel('Maximum Sustained 10-m Wind (kts)')

   plt.grid(True)

   if int(cycle[0:8]) <= 20220924: 
       plt.legend(loc="upper left")
   elif int(cycle[0:8]) >= 20220925: 
       plt.legend(loc="upper right")

   titlestr = 'Hurricane '+TC_name+' Maximum 10-m Wind Traces \n'+ \
              cycle_date.strftime('%HZ %d %b Initializations')+' valid through '+final_valid_date.strftime('%HZ %d %b %Y')
   plt.title(titlestr, fontweight='bold')

   fname = str.lower(TC_name)+'_vmaxtrace_'+cycle

   plt.savefig(OUT_DIR+'/'+fname+'.png',bbox_inches='tight')
   plt.close()




def speedtrace():

   print('plotting speed traces')

   print(len(plot_ospeed))    # same length as date_list

   fig = plt.figure(figsize=(9,8))

   for i in range(len(plot_mspeed)):
      label_str = str.upper(mnames[i])
      if str.upper(mnames[i]) == 'GFS':
         label_str = 'GFS'
      elif str.upper(mnames[i]) == 'GFSO':
         label_str = 'GFSv14'
      elif str.upper(mnames[i]) == 'GEFSMEAN':
         label_str = 'GEFS Mean'
      elif str.upper(mnames[i]) == 'ECMEAN':
         label_str = 'EC Mean'
      elif str.upper(mnames[i]) == 'EMX' or str.upper(mnames[i]) == 'ECMO':
         label_str = 'EC'
      elif str.upper(mnames[i]) == 'UKMET' or str.upper(mnames[i]) == 'UKX' or str.upper(mnames[i]) == 'UK':
         label_str = 'UKM'
      elif str.upper(mnames[i]) == 'HF3A':
         label_str = 'HAFS v0.3A'
      elif str.upper(mnames[i]) == 'HF3S':
         label_str = 'HAFS v0.3S'

      if str.upper(mnames[i][-4:]) == 'MEAN':
      #  plt.plot(plot_mspeed[i], '--', color=line_colors[i], label=label_str, linewidth=2.)
         plt.plot(smooth(plot_mspeed[i],3), '--', color=line_colors[i], label=label_str, linewidth=2.)
      elif str.upper(mnames[i][0:2]) == 'UK':
       # filled = interpolate_gaps(plot_mspeed[i], limit=2)
       # plt.plot(filled, '-', color=line_colors[i], label=label_str, linewidth=2.)
         try:
             filled = interpolate_gaps(plot_mspeed[i][2:], limit=2)
             plt.plot(smooth(filled,2), '-', color=line_colors[i], label=label_str, linewidth=2.)
         except:
             plt.plot(plot_mspeed[i], '-', color=line_colors[i], label=label_str, linewidth=2.)
      else:
       # plt.plot(plot_mspeed[i], '-', color=line_colors[i], label=label_str, linewidth=2.)
         plt.plot(smooth(plot_mspeed[i],3), '-', color=line_colors[i], label=label_str, linewidth=2.)

 # plt.plot(plot_ospeed, '-', color='#696969', label='BEST', linewidth=2., marker='D', markersize=7, markevery=markers_on)
   plt.plot(smooth(plot_ospeed,2), '-', color='#696969', label='BEST', linewidth=3., marker='D', markersize=7, markevery=markers_on)


   xlen = len(plot_ospeed) 
   x = np.arange(0,xlen,1)

   plt.axis([0,xlen,0,20])
   plt.axhspan(0, 20, facecolor='0.5', alpha=0.5)

   labels = []
   for x in range(0,xlen,2):
      if x%4 == 0:
         labels.append(date_list[x].strftime('%HZ %m/%d'))
      else:
         labels.append('')

#  labels = [date_list[x].strftime('%HZ %m/%d') for x in range(0,xlen,6)]
   plt.xticks(np.arange(0,xlen,step=2),labels,rotation=-45,ha='left')
   plt.ylabel('Forward Speed (kts)')

   plt.grid(True)

#  plt.legend(loc="upper left")

   titlestr = 'Hurricane '+TC_name+' Forward Speed \n'+ \
              cycle_date.strftime('%HZ %d %b Initializations')+' valid through '+final_valid_date.strftime('%HZ %d %b %Y')
   plt.title(titlestr, fontweight='bold')

   fname = str.lower(TC_name)+'_speedtrace_'+cycle

   plt.savefig(OUT_DIR+'/'+fname+'.png',bbox_inches='tight')
   plt.close()




def plot_lows(plot):

   thing = np.asarray(plot)
   domain = thing[0]
   fhr = int(thing[1])

   plot_ind = int(fhr/6)

   print('plotting '+cycle+' F'+str(fhr).zfill(2)+' '+TC_name+' on '+domain+' domain')


   print('plotting '+cycle+' '+TC_name+' on '+domain+' domain')

   fig = plt.figure(figsize=(10.9,8.9))
   gs = GridSpec(1,1,wspace=0.0,hspace=0.0)

   # Define where Cartopy maps are located
   cartopy.config['data_dir'] = '/lfs/h2/emc/vpppg/save/'+os.environ['USER']+'/python/NaturalEarth'

   back_res='50m'
   back_img='off'


   if str.upper(domain) == 'CONUS':
       llcrnrlon = -121.5 
       llcrnrlat = 22.
       urcrnrlon = -64.5
       urcrnrlat = 48.
       proj = 'lcc'
       lon0 =-101.
       lat0 = 39.
   elif str.upper(domain) == 'SE_COAST':
       llcrnrlon = -90.
       llcrnrlat = 22.    # 24.
       urcrnrlon = -72.   # -62.5, -67.5
       urcrnrlat = 37.5   # 42
       proj = 'lcc'
       lon0 =-81.
       lat0 = 39.
       standard_parallels=(32,46)
       draw_counties = False
   elif str.upper(domain) == 'FLORIDA':
       llcrnrlon = -87.5
       llcrnrlat = 22.    # 24.
       urcrnrlon = -77.5   # -62.5, -67.5
       urcrnrlat = 32.5   # 42
       proj = 'lcc'
       lon0 =-81.
       lat0 = 39.
       standard_parallels=(32,46)
       draw_counties = True


   extent = [llcrnrlon-1,urcrnrlon+1,llcrnrlat-1,urcrnrlat+1]

   if proj == 'lcc':
       myproj = ccrs.LambertConformal(central_longitude=lon0,central_latitude=lat0,
                false_easting=0.0,false_northing=0.0,secant_latitudes=None,
                standard_parallels=standard_parallels,globe=None)
   elif proj == 'merc':
       myproj = ccrs.Mercator(central_longitude=lon0,min_latitude=llcrnrlat,
                max_latitude=urcrnrlat,latitude_true_scale=lat_ts,globe=None)

   # All lat lons are earth relative, so setup the associated projection correct for that data
   transform = ccrs.PlateCarree()

   ax = fig.add_subplot(gs[0:1,0:1], projection=myproj)
   ax.set_extent(extent,crs=transform)
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
   ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidths=0.5,linestyle='solid',edgecolor='k',zorder=4)
   ax.add_feature(cfeature.STATES.with_scale('50m'),linewidths=0.3,linestyle='solid',edgecolor='k',zorder=4)

   latlongrid = 10.
   if draw_counties:
      ax.add_feature(COUNTIES,facecolor='none',edgecolor='gray')




   for i in range(len(plot_mlats)):

      if str.upper(mnames[i][0:2]) == 'UK' and fhr <= 144 and fhr%12 != 6:
         plot_ind = int(fhr/12) 
      elif str.upper(mnames[i][0:2]) != 'UK': 
         plot_ind = int(fhr/6)

      try:
          if str.upper(mnames[i][-4:]) == 'MEAN':
              ax.text(plot_mlons[i][plot_ind],plot_mlats[i][plot_ind],'L',fontsize=26,fontweight='bold',family='serif',ha='center',va='center',color=line_colors[i],zorder=10,transform=transform)
          elif str.upper(mnames[i][0:2]) == 'UK' and fhr <= 144 and fhr%12 != 6:
              ax.text(plot_mlons[i][plot_ind],plot_mlats[i][plot_ind],'L',fontsize=26,fontweight='bold',ha='center',va='center',color=line_colors[i],zorder=10,transform=transform)
          elif str.upper(mnames[i][0:2]) != 'UK':
              ax.text(plot_mlons[i][plot_ind],plot_mlats[i][plot_ind],'L',fontsize=26,fontweight='bold',ha='center',va='center',color=line_colors[i],zorder=10,transform=transform)
      except:
          print(mnames[i], ' not available for this cycle/fhr')
        # print("plot ind: ", plot_ind)
        # print(len(mtimes[i]))
        # print(len(mlats[i]))

    # if str.upper(mnames[i]) == 'UK':
    #    if fhr > 144 or fhr%12 == 6:
    #       sys.exc_clear() 


   try:
#      x, y = m(olon[plot_ind],olat[plot_ind])
       ax.text(olon[plot_ind],olat[plot_ind],'L',fontsize=26,fontweight='bold',ha='center',va='center',color='#696969',zorder=10,transform=transform)
   except:
       sys.exit('Best Track has ended. Exiting function')



   titlestr1 = 'Hurricane '+TC_name+' Forecast Lows'
   titlestr2 = cycle_date.strftime('Init: %HZ %d %b %Y')+' | '+date_list[plot_ind].strftime('Valid: %HZ %d %b %Y')+' (F'+str(fhr).zfill(2)+')'
   plt.text(0.5, 1.05, titlestr1, fontweight='bold', horizontalalignment='center', transform=ax.transAxes)
   plt.text(0.5, 1.01, titlestr2, fontweight='bold', horizontalalignment='center', transform=ax.transAxes)

   fname = str.lower(TC_name)+'_lows_'+str.lower(domain)+'_'+cycle+'_f'+str(fhr).zfill(3)

   plt.savefig(LOWS_DIR+'/'+fname+'.png',bbox_inches='tight')
   plt.close()




def main():

    # plot composite pressure traces
#   ptrace()

    # Plote composite wind traces
#   vtrace()

    # Plote forward speed traces
#   speedtrace()

#   domains = ['NATL','NW_ATL','SE_COAST','CARIB','GR_ANT']
#   domains = ['HENRI','BOX-NYC']
#   domains = ['GOM','IDA','BARRY','LOUISIANA']
    domains = ['IAN','SE_COAST','GOM']

    pool = multiprocessing.Pool(len(domains))
#   pool.map(plot_tracks,domains)


#   if cycle == '2019083000' or cycle == '2019083012' or cycle == '2019083100':

#       domains = ['SE_COAST','GR_ANT','BAHAMAS']

    domains = ['FLORIDA','SE_COAST']
    fhrs = np.arange(0,121,6)
    plots = [n for n in itertools.product(domains,fhrs)]
    print(plots)
 
    if not os.path.exists(LOWS_DIR):
        os.makedirs(LOWS_DIR)

    for fhr in fhrs:
        for domain in domains:
            plot_lows([domain,fhr])

    exit()


    if cycle[-2:] == '00' or cycle[-2:] == '12':
        domains = ['MIDATL']
        plots = [n for n in itertools.product(domains,fhrs)]
        print(plots)
 
        if not os.path.exists(LOWS_DIR):
            os.makedirs(LOWS_DIR)

        if cycle == '2021083000':
            fhrs = [72]
        elif cycle == '2021083012':
            fhrs = [60]
        elif cycle == '2021083100':
            fhrs = [48]

        pool2 = multiprocessing.Pool(len(plots))
    #   pool2.map(plot_lows,plots)
        for fhr in fhrs:
            for domain in domains:
                plot_lows([domain,fhr])


main()


#plot_lows(['HENRI',24])


