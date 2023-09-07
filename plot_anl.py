#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits
mpl_toolkits.__path__.append('/gpfs/dell2/emc/modeling/noscrub/gwv/py/lib/python/basemap-1.2.1-py3.6-linux-x86_64.egg/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import colors as c
#from pylab import *
import numpy as np
import math
import pygrib, datetime, time, os, sys, subprocess
import multiprocessing, itertools, collections
import scipy, ncepy
from ncepgrib2 import Grib2Encode, Grib2Decode
import dawsonpy
from netCDF4 import Dataset

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
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
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
case = 'westcoast'


# Get machine
machine, hostname = dawsonpy.get_machine()


# Set up working directory
if machine == 'WCOSS':
    pass
elif machine == 'WCOSS_C':
    pass
elif machine == 'WCOSS_DELL_P3':
    DATA_DIR = os.path.join('/gpfs/dell2/emc/verification/noscrub/Logan.Dawson','MEG','data')
    GRAPHX_DIR = os.path.join('/gpfs/dell2/ptmp/Logan.Dawson/MEG/', case, 'graphx')



if os.path.exists(DATA_DIR):
   if not os.path.exists(GRAPHX_DIR):
      os.makedirs(GRAPHX_DIR)
else:
   raise NameError('data for '+case+' case not found')



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
   anl_str = str(sys.argv[2])
except IndexError:
   anl_str = None

if anl_str is None:
   print('Analysis string options: GFS, RAP, MRMS')
   anl_str = raw_input('Enter desired analysis: ')


OUT_DIR = os.path.join(GRAPHX_DIR, anl_str)
if not os.path.exists(OUT_DIR):
      os.makedirs(OUT_DIR)

model_analyses = ['GFS','NAM','NAM3','RAP','HRRR','FV3GFS','EC','ECMWF']
accum_analyses = ['ST4','NOHRSC']
HiResModels = ['NAM3','HIRE','HRRR','HREF']

large_domains = ['CONUS','Australia']
regional_domains = ['eCONUS','eCONUSxE','scCONUS','MIDATLxNE','westcoast']
subregional_domains = ['SPL','MIDSOUTH','Barry']
state_domains = ['OK','Louisiana','DC-NYC','BOX-NYC','WA-BC','Seattle']


### By default, will ask for command line input to determine which analysis files to pull 
### User can uncomment and modify the next line to bypass the command line calls
#nhrs = np.arange(0,19,1)
#nhrs=[0]
#nhrs=np.arange(6,37,6)

#nhrs=[0]

try:
   nhrs
except NameError:
   nhrs = None

if nhrs is None:
   hrb = int(input('Enter first hour (cannot be < 0): '))

   hre = int(input('Enter last hour: '))

   step = int(input('Enter hourly step: '))

   nhrs = np.arange(hrb,hre+1,step)


print('Array of hours is: ')
print(nhrs)

date_list = [date_str + datetime.timedelta(hours=int(x)) for x in nhrs]
#date_list = [date_str + datetime.timedelta(minutes=x) for x in nhrs]

fhrs=[0]

#Specify plots to make
#domains = ['MIDATL','Florence']
domains = ['CONUS','MIDSOUTH']
domains = ['DC-NYC']
#domains = ['OV']
domains = ['CONUS']

if case == 'eastcoast':
    domains = ['CONUS','eCONUSxE','BOX-NYC']
    domains = ['BOX-NYC']
elif case == 'westcoast':
    domains = ['CONUS','westcoast','WA-BC','Seattle']


fields = ['500hght_wind']

if str.upper(anl_str) == 'MRMS':
    fields = ['refc']
elif str.upper(anl_str) == 'ST4':
    fields = ['qpe']
elif str.upper(anl_str) == 'URMA' or str.upper(anl_str) == 'RTMA':
    fields = ['t2_10mwind','td2_10mwind']
    fields = ['10mwind']
elif str.upper(anl_str) == 'GFS' or str.upper(anl_str[0:2]) == 'EC':
    fields = ['wind500','vort500','500hght_slp_10mwind','10mwind']
elif str.upper(anl_str) == 'RAP':
#   fields = ['mlcape_mlcin_shear06']
#   fields = ['slp_mucape_shear06','slp_sbcape_shear06','mlcape_mlcin']
    fields = ['sbcape_sbcin']


#for nfields,ndomains in itertools.product(fields,domains):
plots = [n for n in itertools.product(domains,fields)]
print(plots)

#nplots = [zip(x,fields) for x in itertools.permutations(domains,len(fields))]
#plots = list(itertools.chain(*nplots))
#plots = [y for x in nplots for y in x]
#print len(plots)




def main():

   print(plots)
   pool = multiprocessing.Pool(len(plots))
   pool.map(plot_fields,plots)



def plot_fields(plot):

   print(plot)
   thing = np.asarray(plot)
   domain = thing[0]
   field = thing[1]
#  print thing
#  print domain, field 

   print('plotting '+str.upper(anl_str)+' '+str.upper(field)+' on '+domain+' domain')

   # create figure and axes instances
   if domain == 'CONUS':
    # fig = plt.figure(figsize=(6.9,4.9))
      fig = plt.figure(figsize=(10.9,8.9))
   elif domain == 'SRxSE':
      fig = plt.figure(figsize=(6.9,4.75))
   else:
   #  fig = plt.figure(figsize=(8,8))
      fig = plt.figure(figsize=(11,11))
   ax = fig.add_axes([0.1,0.1,0.8,0.8])


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

   elif domain == 'SRxSE':
      m = Basemap(llcrnrlon=-105,llcrnrlat=22,urcrnrlon=-70,urcrnrlat=40.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-90.,area_thresh=1000.,ax=ax)

   elif domain == 'MIDATLxNE':
      m = Basemap(llcrnrlon=-85,llcrnrlat=35.,urcrnrlon=-65,urcrnrlat=48.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-77.5,area_thresh=1000.,ax=ax)

   elif domain == 'MIDATL':
    # m = Basemap(llcrnrlon=-90.,llcrnrlat=32.5,urcrnrlon=-70.,urcrnrlat=45.,\
      m = Basemap(llcrnrlon=-87.5,llcrnrlat=33.,urcrnrlon=-68.,urcrnrlat=44.,\
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

   elif str.upper(domain) == 'SPL':
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

   barb_length = 5.5
   if domain in large_domains or domain in regional_domains:
      latlongrid = 10.
      if domain in large_domains:
          barb_length = 4.5
      parallels = np.arange(-90.,91.,latlongrid)
      m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
      meridians = np.arange(0.,360.,latlongrid)
      m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
   else:
      m.drawcounties()
      latlongrid = 5.



   if str.upper(anl_str) == 'GFS' or str.upper(anl_str[0:2]) == 'EC':
      HLwindow = 25
      HLfont = 15
      if domain in large_domains:
         skip = 10
      elif domain in regional_domains:
         skip = 6
      elif domain in subregional_domains:
         skip = 4
      elif domain in state_domains:
         skip = 2

   elif str.upper(anl_str) == 'RAP':
      HLwindow = 100
      HLwindow = 50
      skip = 11
      if domain in large_domains:
         skip = 15
      elif domain in subregional_domains:
         skip = 7
      elif domain in state_domains:
         skip = 4

   elif str.upper(anl_str) == 'RTMA' or str.upper(anl_str) == 'URMA':
      HLwindow = 400
      if domain in large_domains:
         skip = 75
      elif domain in regional_domains:
         skip = 60
      elif domain in subregional_domains:
         skip = 30
      elif domain in state_domains:
         skip = 20


   # Sea level pressure
   if str.lower(field) == 'slp':
      print('plotting '+str.upper(anl_str)+' '+str.upper(field)+' analysis')


   # SLP and CAPE
   elif str.lower(field[0:3]) == 'slp' and str.lower(field[6:10]) == 'cape':
      print('plotting '+str.upper(anl_str)+' '+str.upper(field)+' analysis')

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
      if str.upper(anl_str[0:4]) in HiResModels:
         fill_var = scipy.ndimage.gaussian_filter(fill_var,1)

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')
      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('J $\mathregular{kg^{-1}}$')

      contour_var = mslp
      contour_var = scipy.ndimage.gaussian_filter(contour_var,2)
      if str.upper(anl_str[0:4]) in HiResModels:
         contour_var = scipy.ndimage.gaussian_filter(contour_var,2)

      cint  = np.arange(900.,1100.,4.)
      contours = m.contour(lons,lats,contour_var,cint,colors='k',linestyles='solid',latlon=True)
      plt.clabel(contours,cint,colors='k',inline=1,fmt='%.0f',fontsize=9)
      ncepy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow)

      field_title = 'SLP and '+str.upper(field[4:10])

      try:
         if str.lower(field[11:16]) == 'shear':
            print('plotting '+str.upper(anl_str)+' '+str.upper(field[11:18])+' analysis')

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


   # Isotachs
   elif field[0:4] == 'wind':

      if field == 'wind500':
         fill_var = isotach_500
         contour_var = hghts_500
         u_var = uwind_500
         v_var = vwind_500

         clevs = [40,50,60,70,80,90,100]
         clevs = [50,60,70,80,90,100,120]
         colorlist = ['lightsteelblue','skyblue','deepskyblue','dodgerblue','lightpink','fuchsia','darkmagenta']
         cint = np.arange(498.,651.,6.)


    # fill_var = scipy.ndimage.gaussian_filter(fill_var,1)
      contour_var = scipy.ndimage.gaussian_filter(contour_var,1)
      if str.upper(anl_str[0:4]) in HiResModels:
    #    fill_var = scipy.ndimage.gaussian_filter(fill_var,1)
         contour_var = scipy.ndimage.gaussian_filter(contour_var,1)


      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

      contours = m.contour(lons,lats,contour_var,cint,colors='k',linewidths=1.5,latlon=True)
   #  ax.clabel(contours,cint,colors='k',inline=1,fmt='%.f',fontsize=10)
      ax.clabel(contours,colors='k',inline=1,fmt='%d',fontsize=10)

      m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle')

      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('kts')

   #  ncepy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow)
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
      if str.upper(anl_str[0:4]) in HiResModels:
         fill_var = scipy.ndimage.gaussian_filter(fill_var,1)
         contour_var = scipy.ndimage.gaussian_filter(contour_var,1)


      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

      contours = m.contour(lons,lats,contour_var,cint,colors='k',linewidths=1.5,latlon=True)
   #  ax.clabel(contours,cint,colors='k',inline=1,fmt='%.0f',fontsize=10)
      ax.clabel(contours,colors='k',inline=1,fmt='%d',fontsize=10)

      if field[-7:] == 'wind500':

         fill_var2 = isotach_500
         clevs2 = [40,50,60,70,80,90,100]
         colorlist2 = ['lightsteelblue','skyblue','deepskyblue','dodgerblue','lightpink','fuchsia','darkmagenta']

         fill2 = m.contourf(lons,lats,fill_var2,clevs2,latlon=True,colors=colorlist2,extend='max')

         cax1 = fig.add_axes([0.14,0.17,0.3,0.025])
         cbar1 = plt.colorbar(fill,cax=cax1,ticks=clevs,orientation='horizontal')
         cbar1.ax.tick_params(labelsize=10)
         cbar1.set_label('m $\mathregular{s^{-1}}$')

         cax2 = fig.add_axes([0.56,0.17,0.3,0.025])
         cbar2 = plt.colorbar(fill2,cax=cax2,ticks=clevs2,orientation='horizontal')
         cbar2.ax.tick_params(labelsize=10)
         cbar2.set_label('$\mathregular{10^{-5} s^{-1}}$')


      else:

          m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle',color='steelblue')

          cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
          cbar.ax.tick_params(labelsize=10)
          cbar.set_label('$\mathregular{x10^{-5}}$ $\mathregular{s^{-1}}$')


  #   ncepy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow)
      dawsonpy.plt_highs_and_lows(m,mslp,lons,lats,mode='reflect',window=HLwindow,font=HLfont)

#     for label in cbar.ax.xaxis.get_ticklabels():
#        label.set_visible(False)
#     for label in cbar.ax.xaxis.get_ticklabels()[::5]:
#        label.set_visible(True)

      field_title = field[4:7]+' mb Heights, Vorticity, Winds, and Surface Lows/Highs'


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
      if str.upper(anl_str[0:4]) in HiResModels:
         contour_var = scipy.ndimage.gaussian_filter(contour_var,2)

      cint  = np.arange(900.,1100.,4.)
      contours = m.contour(lons,lats,contour_var,cint,colors='k',linestyles='solid',latlon=True)
      ax.clabel(contours,colors='k',inline=1,fmt='%.0f',fontsize=9)
     #ax.clabel(contours,cint,colors='k',inline=1,fmt='%d',fontsize=10)
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



   # Temperature
   elif field[0:2] == 't2':
      clevs = np.arange(-16,132,4)
      tlevs = [str(clev) for clev in clevs]

      colormap = cmap_t2m()
      norm = matplotlib.colors.BoundaryNorm(clevs, colormap.N)

      fill_var = t2

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,cmap=colormap,norm=norm,extend='both')

    # cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.5,aspect=15)
      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.set_xticklabels(tlevs)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label(u'\xb0''F')

      for label in cbar.ax.xaxis.get_ticklabels():
         label.set_visible(False)
      for label in cbar.ax.xaxis.get_ticklabels()[::4]:
         label.set_visible(True)

      field_title = '2-m Temperature'


      try:
         if str.lower(field[3:10]) == '10mwind':

            u_var = u10
            v_var = v10

            m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle')

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


   # CAPE / CIN / SHEAR
   elif str.lower(field[2:6]) == 'cape':
      print('plotting '+str.upper(anl_str)+' '+str.upper(field[0:6])+' analysis')

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
      if str.upper(anl_str[0:4]) in HiResModels:
         fill_var = scipy.ndimage.gaussian_filter(fill_var,1)

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('J $\mathregular{kg^{-1}}$')

      field_title = str.upper(field[0:6])

      try:
         if str.lower(field[9:12]) == 'cin':
            print('plotting '+str.upper(anl_str)+' '+str.upper(field[7:12])+' analysis')
 
            if str.lower(field[0:2]) == 'sb':
               contour_var = sbcin
            elif str.lower(field[0:2]) == 'ml':
               contour_var = mlcin

            contour_var = scipy.ndimage.gaussian_filter(contour_var,2)
            if str.upper(anl_str[0:4]) in HiResModels:
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
            print('plotting '+str.upper(anl_str)+' '+str.upper(field[13:20])+' analysis')

            if field[18] == '0' and field[19] == '6':
               u_var = u_shr06
               v_var = v_shr06
            elif field[18] == '0' and field[19] == '1':
               u_var = u_shr01
               v_var = v_shr01

            m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=4.5,sizes={'spacing':0.2},pivot='middle')

            field_title = str.upper(field[0:6])+', '+str.upper(field[7:12])+', and '+field[18]+'-'+field[19]+' km '+str.title(field[13:18])

      except IndexError:
         sys.exc_clear()


   # Simulated reflectivity
   elif field[0:3] == 'ref':
      clevs = np.linspace(5,70,14)
      colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green', \
                   'yellow','gold','darkorange','red','firebrick','darkred','fuchsia','darkmagenta']

      if field == 'refc':
         fill_var = refc
      elif field == 'refd1':
         fill_var = refd1
      elif field == 'refd4':
         fill_var = refd4
      
    # fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')
      xlons, ylats = m(lons,lats)
      fill = m.contourf(xlons,ylats,fill_var,clevs,colors=colorlist,extend='max')

      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('dBZ')

      if field == 'refc':
         field_title = 'Composite Reflectivity'
      elif field == 'refd1':
         field_title = 'Seamless Hybrid Scan Reflectivity'
      elif field[0:4] == 'refd':
         field_title = field[4]+'-km AGL Reflectivity'


   # QPE
   elif str.lower(field) == 'qpe':
      print('plotting '+str.upper(anl_str)+' '+str.upper(field)+' analysis')

      clevs = [0.01,0.1,0.25,0.5,0.75,1,1.5,2,2.5,3,4,5,6,8,10,12,14,16,18,20]
      tlevs = [str(clev) for clev in clevs]
      colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','cyan','slateblue', \
                   'mediumorchid','darkmagenta','darkred','crimson','darkorange','salmon',  \
                   'yellow','saddlebrown','magenta','pink','beige','black']

      fill_var = precip

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.set_xticklabels(tlevs)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('in')

      field_title = str(nhrs[j])+'-h '+str.upper(field)
      field = str(nhrs[j])+'h'+field


   # Snow depth
   elif str.lower(field[0:4]) == 'snod':
      print('plotting '+str.upper(anl_str)+' '+str.upper(field)+' analysis')

#     clevs = [0.1,1.,2.,4.,6.,8.,10.,12.,15.,20.,25.]
#     colorlist = ['slateblue','deepskyblue','blue','lawngreen','green', \
#                  'khaki','yellow','lightsalmon','darkorange','red','darkred']
      clevs = [0.1,1,2,3,4,5,6,8,12,18,24,30,36,48]
      colorlist = ['powderblue','lightsteelblue','cornflowerblue','steelblue', \
                   'royalblue','blue', \
                   'khaki','orange','darkorange','red', \
                   'firebrick','darkred','maroon','indigo']

      fill_var = snow

      fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')

      cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
      cbar.ax.tick_params(labelsize=10)
      cbar.set_label('in')

      field_title = str(nhrs[j])+'-h Snowfall'
      field = str(nhrs[j])+'h'+field


   #################################
   ## TITLE AND FILENAME SETTINGS ##
   #################################

   if str.upper(anl_str) == 'NAM3':
      mod_str = str.upper(anl_str[0:3])+' Nest Analysis'
   elif str.upper(anl_str[0:6]) == 'HIRESW':
      mod_str = anl_str[0:6]+' '+str.upper(anl_str[6:])
   elif str.lower(anl_str[4:]) == 'mean':
      mod_str = str.upper(anl_str[0:4])+' Mean'
   elif str.upper(anl_str) == 'MRMS':
      mod_str = str.upper(anl_str)
   elif str.upper(anl_str) == 'ST4':
      mod_str = 'Stage IV Analysis'
   else:
      mod_str = str.upper(anl_str)+ ' Analysis'
   var_str = field_title
   initstr = date_str.strftime('Init: %HZ %d %b %Y')
   if str.upper(anl_str) in accum_analyses:
      validstr = date_str.strftime('Valid: %HZ %d %b to ')+date_list[j].strftime('%HZ %d %b %Y')
   elif str.upper(anl_str) in model_analyses:
      validstr = date_list[j].strftime('Valid: %HZ %d %b %Y')+' (F'+str(fhrs[0]).zfill(digits)+')'
   else:
      validstr = date_list[j].strftime('Valid: %HZ %d %b %Y')
   plt.text(0, 1.06, mod_str, horizontalalignment='left', transform=ax.transAxes, fontweight='bold')
   plt.text(0, 1.01, var_str, horizontalalignment='left', transform=ax.transAxes, fontweight='bold')
   if fhrs[0] != 0:
      plt.text(1, 1.06, initstr, horizontalalignment='right', transform=ax.transAxes, fontweight='bold')
   plt.text(1, 1.01, validstr, horizontalalignment='right', transform=ax.transAxes, fontweight='bold')


   fname = str.lower(field)+'_'+str.lower(domain)+'_'+YYYYMMDDCC

   plt.savefig(OUT_DIR+'/'+fname+'.png',bbox_inches='tight')
   plt.close()



#Loop to plot
for j in range(len(date_list)):
#for j in range(1):
   YYYYMMDDCC = date_list[j].strftime("%Y%m%d%H")
   print(nhrs[j])
   print(date_list[j])

   if str.upper(anl_str) == 'GFS' or str.upper(anl_str) == 'GEFS' or str.upper(anl_str) == 'EC':
      digits = 3
   else:
      digits = 2

   # GFS
   if str.upper(anl_str) == 'GFS':
      try:
         fil = '/gpfs/hps/nco/ops/com/gfs/prod/gfs.'+YYYYMMDDCC[0:8]+'/'+ \
               str.lower(anl_str)+'.t'+YYYYMMDDCC[8:10]+'z.pgrb2.0p25.f000'
         grbs = pygrib.open(fil)
      except:   
         fil = DATA_DIR+'/'+str.lower(anl_str)+'.'+YYYYMMDDCC[0:8]+\
               '/gfs.t'+YYYYMMDDCC[8:10]+'z.pgrb2.0p25.f000'
         grbs = pygrib.open(fil)


#     td2 = grbs.select(stepType='instant',name='2 metre dewpoint temperature',typeOfLevel='heightAboveGround',level=2)[0].values*9/5 - 459.67
#     u10 = grbs.select(stepType='instant',name='10 metre U wind component',typeOfLevel='heightAboveGround',level=10)[0].values*1.94384
#     v10 = grbs.select(stepType='instant',name='10 metre V wind component',typeOfLevel='heightAboveGround',level=10)[0].values*1.94384

   # EC
   elif str.upper(anl_str[0:2]) == 'EC':
      try:
         fil = '/gpfs/hps/nco/ops/com/gfs/prod/gfs.'+YYYYMMDDCC[0:8]+'/'+ \
               'ecmwf.t'+YYYYMMDDCC[8:10]+'z.pgrb2.0p25.f000'
         grbs = pygrib.open(fil)
      except:   
         fil = DATA_DIR+'/ecmwf.'+YYYYMMDDCC[0:8]+\
               '/ec.t'+YYYYMMDDCC[8:10]+'z.0p25.f000'
         print(fil)
         grbs = pygrib.open(fil)



   # NAM Nest
   elif str.upper(anl_str) == 'NAM3':
      try:
         fil = '/com2/nam/prod/nam.'+YYYYMMDDCC[0:8]+'/'+str.lower(anl_str[0:3])+'.t'+ \
               YYYYMMDDCC[8:10]+'z.conusnest.hiresf00.tm00.grib2'
         grbs = pygrib.open(fil)
      except:   
         fil = DATA_DIR+'/'+str.lower(anl_str[0:3])+'.'+YYYYMMDDCC[0:8]+'.t'+ \
               YYYYMMDDCC[8:10]+'z.conusnest.hiresf00.tm00.grib2'
         grbs = pygrib.open(fil)


   # RAP
   elif str.upper(anl_str[0:3]) == 'RAP':
      try:
         fil = '/gpfs/hps/nco/ops/com/rap/prod/rap.'+YYYYMMDDCC[0:8]+'/'+str.lower(anl_str[0:6])+'.t'+ \
               YYYYMMDDCC[8:10]+'z.awp130pgrbf00.grib2'
         grbs = pygrib.open(fil)
      except:   
         fil = DATA_DIR+'/'+str.lower(anl_str[0:6])+'.'+YYYYMMDDCC[0:8]+'.t'+ \
               YYYYMMDDCC[8:10]+'z.awp130pgrbf00.grib2'
         grbs = pygrib.open(fil)



   # HRRR
   elif str.upper(anl_str[0:4]) == 'HRRR':
      try:
         fil = '/gpfs/hps/nco/ops/com/hrrr/prod/hrrr.'+YYYYMMDDCC[0:8]+'/conus/'+str.lower(anl_str[0:6])+'.t'+ \
               YYYYMMDDCC[8:10]+'z.wrfprsf00.grib2'
         grbs = pygrib.open(fil)
      except:   
         fil = DATA_DIR+'/'+str.lower(anl_str[0:6])+'.'+YYYYMMDDCC[0:8]+'.t'+ \
               YYYYMMDDCC[8:10]+'z.wrfprsf00.grib2'
         grbs = pygrib.open(fil)


   # RTMA/URMA
   elif str.upper(anl_str) == 'RTMA' or str.upper(anl_str) == 'URMA':
      try:
         fil = '/com2/'+str.lower(anl_str)+'/prod/'+str.lower(anl_str)+'2p5.'+YYYYMMDDCC[0:8]+'/'+ \
               str.lower(anl_str)+'2p5.t'+YYYYMMDDCC[8:10]+'z.2dvaranl_ndfd.grb2'
         grbs = pygrib.open(fil)
      except:
         fil = DATA_DIR+'/'+str.lower(anl_str)+'2p5.'+YYYYMMDDCC[0:8]+'/'+ \
               str.lower(anl_str)+'2p5.t'+YYYYMMDDCC[8:10]+'z.2dvaranl_ndfd.grb2_wexp'
         grbs = pygrib.open(fil)

      u10 = grbs.select(stepType='instant',name='10 metre U wind component',typeOfLevel='heightAboveGround',level=10)[0].values*1.94384
      v10 = grbs.select(stepType='instant',name='10 metre V wind component',typeOfLevel='heightAboveGround',level=10)[0].values*1.94384
      if '10mwind' in fields:
          isotach_10m = np.sqrt(u10**2+v10**2)

      # Temperature in Fahrenheit
      grb = grbs.select(stepType='instant',name='2 metre temperature',typeOfLevel='heightAboveGround',level=2)[0]
      t2  = grb.values*9/5 - 459.67
      td2 = grbs.select(stepType='instant',name='2 metre dewpoint temperature',typeOfLevel='heightAboveGround',level=2)[0].values*9/5 - 459.67



   # MRMS
   elif str.upper(anl_str) == 'MRMS':

      if 'refc' in fields:
         try:
            fil = DATA_DIR+'/MergedReflectivityQCComposite_00.50_'+ \
                  YYYYMMDDCC[0:8]+'-'+YYYYMMDDCC[8:10]+'0000.nc'
     #      fil = '/com/hourly/prod/radar.'+YYYYMMDDCC[0:8]+'/'+ \
     #            'refd3d.t'+YYYYMMDDCC[8:10]+'z.grb2f00'
            nc   = Dataset(fil)
            refc = nc.variables['VAR_209_10_0_P0_L102_GLL0'][:]
            lat  = nc.variables['lat_0'][:]
            lon  = nc.variables['lon_0'][:]

            lons, lats = np.meshgrid(lon,lat)

         except:   
            fil = DATA_DIR+'/refd3d.'+YYYYMMDDCC[0:8]+'.t'+ \
                  YYYYMMDDCC[8:10]+'z.grb2f00'
            grbs = pygrib.open(fil)


       # grb = grbs.select(name='Maximum/Composite radar reflectivity')[0]
       # grb = grbs.message(1)
       # refc = grb.values

      if 'refd1' in fields:
         try:
            fil = '/dcom/us007003/ldmdata/obs/upperair/mrms/conus'+ \
                  'refd3d.t'+YYYYMMDDCC[8:10]+'z.grb2f00'
            grbs = pygrib.open(fil)
         except:   
            fil = DATA_DIR+'/MRMS/SeamlessHSR_00.00_'+YYYYMMDDCC[0:8]+ \
                  '-'+YYYYMMDDCC[8:10]+'0000.grib2'
            grbs = pygrib.open(fil)


         grbs.seek(0)
         for gr in grbs:
             print(gr.name)
         grb = grbs.select(name='unknown')[0]
       # refd1 = grbs.message(1)
       # refd1 = grb.values




   # Stage IV
   elif str.upper(anl_str) == 'ST4':
      try:
         fil = '/gpfs/hps/nco/ops/com/hrrr/prod/hrrr.'+cycle[0:8]+'/conus/'+str.lower(anl_str[0:6])+'.t'+ \
               cycle[8:10]+'z.wrfprsf00.grib2'
         grbs = pygrib.open(fil)
      except:   
         fil = DATA_DIR+'/'+str.upper(anl_str)+'.'+YYYYMMDDCC+'.06h'
         grbs = pygrib.open(fil)

         grb = grbs.select(name='Total Precipitation')[0]

         if nhrs[j] == 6:
            precip = grb.values/25.4
         else:
            bucket = grb.values/25.4
            precip = precip + bucket


   # NOHRSC
   elif str.upper(anl_str) == 'NOHRSC':
      try:
         fil = '/gpfs/hps/nco/ops/com/hrrr/prod/hrrr.'+cycle[0:8]+'/conus/'+str.lower(anl_str[0:6])+'.t'+ \
               cycle[8:10]+'z.wrfprsf00.grib2'
         grbs = pygrib.open(fil)
      except:   
         fil = DATA_DIR+'/sfav2_CONUS_6h_'+YYYYMMDDCC+'_grid184.grb2'
         grbs = pygrib.open(fil)

         grb = grbs.select(name='Total snowfall')[0]

         if nhrs[j] == 6:
            snow = grb.values*39.3701
         else:
            bucket = grb.values*39.3701
            snow = snow + bucket

   if str.upper(anl_str) in model_analyses:

      hghts_500  = grbs.select(name='Geopotential Height',typeOfLevel='isobaricInhPa',level=500)[0].values*0.1
      if anl_str == 'GFS':
          vort_500 = grbs.select(name='Absolute vorticity',typeOfLevel='isobaricInhPa',level=500)[0].values*1.e5
      elif anl_str[0:2] == 'EC':
          relv_500 = grbs.select(name='Vorticity (relative)',typeOfLevel='isobaricInhPa',level=500)[0].values
          vort_500 = hghts_500*0.
      uwind_500 = grbs.select(name='U component of wind',typeOfLevel='isobaricInhPa',level=500)[0].values*1.94384
      vwind_500 = grbs.select(name='V component of wind',typeOfLevel='isobaricInhPa',level=500)[0].values*1.94384
      isotach_500 = np.sqrt(uwind_500**2+vwind_500**2)

      u10 = grbs.select(stepType='instant',name='10 metre U wind component')[0].values*1.94384
      v10 = grbs.select(stepType='instant',name='10 metre V wind component')[0].values*1.94384
      if '10mwind' in fields:
          isotach_10m = np.sqrt(u10**2+v10**2)

#     u10 = grbs.select(stepType='instant',name='10 metre U wind component',typeOfLevel='heightAboveGround',level=10)[0].values*1.94384
#     v10 = grbs.select(stepType='instant',name='10 metre V wind component',typeOfLevel='heightAboveGround',level=10)[0].values*1.94384


#     mucape = grbs.select(name='Convective available potential energy',typeOfLevel='pressureFromGroundLayer',topLevel=18000)[0].values
#     mlcape = grbs.select(name='Convective available potential energy',typeOfLevel='pressureFromGroundLayer',topLevel=9000)[0].values
#     mlcin  = grbs.select(name='Convective inhibition',typeOfLevel='pressureFromGroundLayer',topLevel=9000)[0].values
#     sbcape = grbs.select(name='Convective available potential energy',level=0)[0].values
#     sbcin = grbs.select(name='Convective inhibition',level=0)[0].values

#     u_shr06 = grbs.select(name='Vertical u-component shear')[0].values*1.94384
#     v_shr06 = grbs.select(name='Vertical v-component shear')[0].values*1.94384



      if str.upper(anl_str[0:3]) == 'RAP' or str.upper(anl_str[0:4]) == 'HRRR':
         grb = grbs.select(name='MSLP (MAPS System Reduction)')[0]
      elif str.upper(anl_str[0:2]) == 'EC':
         grb = grbs.select(name='Mean sea level pressure')[0]
         print(grb)
      else:
         try:
            grb = grbs.select(name='MSLP (Eta model reduction)')[0]
         except:
            grb = grbs.select(name='Pressure reduced to MSL')[0]
      mslp = grb.values*0.01
      print(grb)

   try:
       lats, lons = grb.latlons()
       grbs.close()
   except:
       pass


   if anl_str[0:2] == 'EC':
#     phi = np.array(lats)
#     sin_phi = math.sin(phi.flatten())
#     f = 2*7.292E-5*np.reshape(sin_phi,lats.shape)
      f = 2*7.292E-5*np.sin(lats*(np.pi/180.))
      vort_500 = (relv_500 + f)*1.e5

#  plot_fields(('DC-NYC', 'refc'))
   main()



