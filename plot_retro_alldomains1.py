import pygrib


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as image
import mpl_toolkits
mpl_toolkits.__path__.append('/gpfs/dell2/emc/modeling/noscrub/gwv/py/lib/python/basemap-1.2.1-py3.6-linux-x86_64.egg/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap

from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap

import numpy as np
import time,os,sys,multiprocessing
import ncepy, dawsonpy
from scipy import ndimage
#from netCDF4 import Dataset
import pyproj

#--------------Define some functions ------------------#

def clear_plotables(ax,keep_ax_lst,fig):
  #### - step to clear off old plottables but leave the map info - ####
  if len(keep_ax_lst) == 0 :
    print("clear_plotables WARNING keep_ax_lst has length 0. Clearing ALL plottables including map info!")
  cur_ax_children = ax.get_children()[:]
  if len(cur_ax_children) > 0:
    for a in cur_ax_children:
      if a not in keep_ax_lst:
       # if the artist isn't part of the initial set up, remove it
        a.remove()

def compress_and_save(filename):
  #### - compress and save the image - ####
  plt.savefig(filename, format='png', bbox_inches='tight', dpi=150)
# ram = cStringIO.StringIO()
# plt.savefig(ram, format='png', bbox_inches='tight', dpi=150)
# ram.seek(0)
# im = Image.open(ram)
# im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
# im2.save(filename, format='PNG')

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


def cmap_t850():
 # Create colormap for 850-mb equivalent potential temperature
    r=np.array([255,128,0,  70, 51, 0,  0,  0, 51, 255,255,255,255,255,171,128,128,96,201])
    g=np.array([0,  0,  0,  70, 102,162,225,92,153,255,214,153,102,0,  0,  0,  68, 96,201])
    b=np.array([255,128,128,255,255,255,162,0, 102,0,  112,0,  0,  0,  56, 0,  68, 96,201])
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
    cmap_t850_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_T850_COLTBL',colorDict)
    return cmap_t850_coltbl


def cmap_terra():
 # Create colormap for terrain height
 # Emerald green to light green to tan to gold to dark red to brown to light brown to white
    r=np.array([0,  152,212,188,127,119,186])
    g=np.array([128,201,208,148,34, 83, 186])
    b=np.array([64, 152,140,0,  34, 64, 186])
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
    cmap_terra_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_TERRA_COLTBL',colorDict)
    cmap_terra_coltbl.set_over(color='#E0EEE0')
    return cmap_terra_coltbl


def extrema(mat,mode='wrap',window=100):
    # find the indices of local extrema (max only) in the input array.
    mx = ndimage.filters.maximum_filter(mat,size=window,mode=mode)
    # (mat == mx) true if pixel is equal to the local max
    return np.nonzero(mat == mx)

#-------------------------------------------------------#

# Necessary to generate figs when not running an Xserver (e.g. via PBS)
plt.switch_backend('agg')

# Read date/time and forecast hour from command line
case = str(sys.argv[1])
cycle = str(sys.argv[2])
ymd = cycle[0:8]
year = int(cycle[0:4])
month = int(cycle[4:6])
day = int(cycle[6:8])
hour = int(cycle[8:10])
cyc = str(hour).zfill(2)
print(year, month, day, hour)

fhr = int(sys.argv[3])
fhour = str(fhr).zfill(2)
print('fhour '+fhour)

# Forecast valid date/time
itime = cycle
vtime = ncepy.ndate(itime,int(fhr))


# Get machine and head directory
machine, hostname = dawsonpy.get_machine()

machine = 'WCOSS_DELL_P3'
if machine == 'WCOSS_DELL_P3':
    DIR = '/gpfs/hps3/emc/meso/noscrub/Matthew.Pyle/fv3_packing_test'


# Make sure prod directory doesn't fail
try:
    PROD_DIR = os.environ['PROD_DIR']
except KeyError:
    PROD_DIR = DIR+'/orig_packing'
    print('PROD_DIR not defined in environment. Setting PROD_DIR to '+PROD_DIR)

# Make sure para directory doesn't fail
try:
    PARA_DIR = os.environ['PARA_DIR']
except KeyError:
    PARA_DIR = DIR+'/new_packing'
    print('PARA_DIR not defined in environment. Setting PARA_DIR to '+PARA_DIR)



# Define the output files
# Prod, para
model_str = str(sys.argv[4])
if str.upper(model_str) == 'HRW-FV3':
    prod_str = str.upper(model_str)+' orig'
    para_str = str.upper(model_str)+' new'
    data1 = pygrib.open(PROD_DIR+'/hiresw.t'+cyc+'z.fv3_5km.f'+fhour+'.conus.grib2')
    data2 = pygrib.open(PARA_DIR+'/hiresw.t'+cyc+'z.fv3_5km.f'+fhour+'.conus.grib2')

# Get the lats and lons
grids = [data1, data2]
lats = []
lons = []
lats_shift = []
lons_shift = []

for data in grids:
    # Unshifted grid for contours and wind barbs
    lat, lon = data[1].latlons()
    lats.append(lat)
    lons.append(lon)

    # Shift grid for pcolormesh
    lat1 = data[1]['latitudeOfFirstGridPointInDegrees']
    lon1 = data[1]['longitudeOfFirstGridPointInDegrees']
    try:
        nx = data[1]['Nx']
        ny = data[1]['Ny']
    except:
        nx = data[1]['Ni']
        ny = data[1]['Nj']

    proj_params = data[1].projparams
    try:
        dx = data[1]['DxInMetres']
        dy = data[1]['DyInMetres']
    except:
        dx = data[1]['iDirectionIncrementInDegrees']
        dy = data[1]['jDirectionIncrementInDegrees']
        proj_params['proj'] = 'latlon'

    pj = pyproj.Proj(proj_params)
    llcrnrx, llcrnry = pj(lon1,lat1)
    llcrnrx = llcrnrx - (dx/2.)
    llcrnry = llcrnry - (dy/2.)
    x = llcrnrx + dx*np.arange(nx)
    y = llcrnry + dy*np.arange(ny)
    x,y = np.meshgrid(x,y)
    lon, lat = pj(x, y, inverse=True)
    lats_shift.append(lat)
    lons_shift.append(lon)

# Unshifted lat/lon arrays grabbed directly using latlons() method
lat = lats[0]
lon = lons[0]
lat2 = lats[1]
lon2 = lons[1]

# Shifted lat/lon arrays for pcolormesh 
lat_shift = lats_shift[0]
lon_shift = lons_shift[0]
lat2_shift = lats_shift[1]
lon2_shift = lons_shift[1]


# Grid settings for wind rotation
Lon0 = data1[1]['LoVInDegrees']
Lat0 = data1[1]['LaDInDegrees']



###################################################
# Read in all variables and calculate differences #
###################################################
t1a = time.clock()



# Specify vars
params = ['HGT','TMP','DPT','UGRD','VGRD','VVEL']
levels = [200,300,400,500,600,700,850,925,1000]
levels = [250,500,850,1000]

hgt_list_1 = []
tmp_list_1 = []
dpt_list_1 = []
ugrd_list_1 = []
vgrd_list_1 = []
vvel_list_1 = []

hgt_list_2 = []
tmp_list_2 = []
dpt_list_2 = []
ugrd_list_2 = []
vgrd_list_2 = []
vvel_list_2 = []


for level in levels:
    hgt_1 = data1.select(name='Geopotential Height',level=level)[0].values
    hgt_2 = data2.select(name='Geopotential Height',level=level)[0].values

    tmp_1 = data1.select(name='Temperature',level=level)[0].values - 273.15
    tmp_2 = data2.select(name='Temperature',level=level)[0].values - 273.15

    dpt_1 = data1.select(name='Dew point temperature',level=level)[0].values - 273.15
    dpt_2 = data2.select(name='Dew point temperature',level=level)[0].values - 273.15

    ugrd_1 = data1.select(name='U component of wind',level=level)[0].values
    ugrd_2 = data2.select(name='U component of wind',level=level)[0].values

    vgrd_1 = data1.select(name='V component of wind',level=level)[0].values
    vgrd_2 = data2.select(name='V component of wind',level=level)[0].values

    ugrd_1, vgrd_1 = ncepy.rotate_wind(Lat0,Lon0,lon,ugrd_1,vgrd_1,'lcc',inverse=False)
    ugrd_2, vgrd_2 = ncepy.rotate_wind(Lat0,Lon0,lon,ugrd_2,vgrd_2,'lcc',inverse=False)

    vvel_1 = data1.select(name='Vertical velocity',level=level)[0].values
    vvel_2 = data2.select(name='Vertical velocity',level=level)[0].values

    hgt_list_1.append(hgt_1)
    hgt_list_2.append(hgt_2)

    tmp_list_1.append(tmp_1)
    tmp_list_2.append(tmp_2)

    dpt_list_1.append(dpt_1)
    dpt_list_2.append(dpt_2)

    ugrd_list_1.append(ugrd_1)
    ugrd_list_2.append(ugrd_2)

    vgrd_list_1.append(vgrd_1)
    vgrd_list_2.append(vgrd_2)

    vvel_list_1.append(vvel_1)
    vvel_list_2.append(vvel_2)


data1.close()
data2.close()


t2a = time.clock()
t3a = round(t2a-t1a, 3)
print(("%.3f seconds to read all messages") % t3a)


# Specify plotting domains
domains = ['conus']

# colors for difference plots, only need to define once
difcolors = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','white','white','#EEEE00','#EEC900','darkorange','orangered','red']

# colors for cloud cover plots
cdccolors = ['#FFFFFF','#F0F0F0','#E0E0E0','#D8D8D8','#C8C8C8','#B8B8B8','#A8A8A8','#909090','#787878','#696969']



########################################
#    START PLOTTING FOR EACH DOMAIN    #
########################################



def main():

  # Number of processes must coincide with the number of domains to plot
  pool = multiprocessing.Pool(len(levels))
  pool.map(plot_all,levels)

def plot_all(level):

  dom = 'conus'
  t1dom = time.clock()
  print('Working on '+dom+' for '+str(level)+' mb')

  # create figure and axes instances
  fig = plt.figure()
  gs = GridSpec(9,9,wspace=0.0,hspace=0.0)
  ax1 = fig.add_subplot(gs[0:4,0:4])
  ax2 = fig.add_subplot(gs[0:4,5:])
  ax3 = fig.add_subplot(gs[5:,1:8])
  axes = [ax1, ax2, ax3]
  if machine == 'WCOSS':
    im = image.imread('/gpfs/hps3/emc/meso/save/Benjamin.Blake/python.raphrrr/noaa.png')
  elif machine == 'WCOSS_C':
    im = image.imread('/gpfs/hps3/emc/meso/save/Benjamin.Blake/python.raphrrr/noaa.png')
  elif machine == 'WCOSS_DELL_P3':
    im = image.imread('/gpfs/dell2/emc/modeling/noscrub/Benjamin.Blake/python.fv3/noaa.png')
  elif machine == 'HERA':
    im = image.imread('/gpfs/dell2/emc/modeling/noscrub/Benjamin.Blake/python.fv3/noaa.png')
  par = 1

  # Map corners for each domain
  if dom == 'conus':
    llcrnrlon = -125.5
    llcrnrlat = 18.0
    urcrnrlon = -57.5
    urcrnrlat = 52.0
    lat_0 = 35.4
    lon_0 = -97.6
    xscale=0.15
    yscale=0.2
  elif dom == 'splains':
    llcrnrlon = -105.0
    llcrnrlat = 28.5
    urcrnrlon = -88.0
    urcrnrlat = 40.
    lat_0 = 33.5
    lon_0 = -96.5
    xscale=0.17
    yscale=0.18
  elif dom == 'nplains':
    llcrnrlon = -105.0
    llcrnrlat = 38.0
    urcrnrlon = -88.0
    urcrnrlat = 49.0
    lat_0 = 33.5
    lon_0 = -96.5
    xscale=0.17
    yscale=0.18
  elif dom == 'cplains':
    llcrnrlon = -105.0
    llcrnrlat = 32.5 
    urcrnrlon = -88.0
    urcrnrlat = 43.5
    lat_0 = 33.5
    lon_0 = -96.5
    xscale=0.17
    yscale=0.18
  elif dom == 'midwest':
    llcrnrlon = -96.5
    llcrnrlat = 36.0
    urcrnrlon = -79.5
    urcrnrlat = 47.5
    lat_0 = 33.5
    lon_0 = -88.0
    xscale=0.17
    yscale=0.18
  elif dom == 'northeast':
    llcrnrlon = -84.5
    llcrnrlat = 36.0
    urcrnrlon = -64.5
    urcrnrlat = 47.5
    lat_0 = 33.5
    lon_0 = -74.5
    xscale=0.17
    yscale=0.18
  elif dom == 'southeast':
    llcrnrlon = -94.0
    llcrnrlat = 26.5
    urcrnrlon = -75.0
    urcrnrlat = 38.0
    lat_0 = 33.5
    lon_0 = -84.5
    xscale=0.17
    yscale=0.18
  elif dom == 'northwest':
    llcrnrlon = -125.0
    llcrnrlat = 37.0 
    urcrnrlon = -102.0
    urcrnrlat = 49.5
    lat_0 = 45.0
    lon_0 = -113.5
    xscale=0.15
    yscale=0.18
  elif dom == 'southwest':
    llcrnrlon = -123.5
    llcrnrlat = 30.0 
    urcrnrlon = -100.5
    urcrnrlat = 42.5
    lat_0 = 37.0
    lon_0 = -112.0
    xscale=0.15
    yscale=0.18
  elif dom == 'BN':
    llcrnrlon = -75.75
    llcrnrlat = 40.0
    urcrnrlon = -69.5
    urcrnrlat = 43.0
    lat_0 = 41.0
    lon_0 = -74.6
    xscale=0.14
    yscale=0.19
  elif dom == 'CE':
    llcrnrlon = -103.0
    llcrnrlat = 32.5
    urcrnrlon = -88.5
    urcrnrlat = 41.5
    lat_0 = 35.0
    lon_0 = -97.0
    xscale=0.15
    yscale=0.18
  elif dom == 'CO':
    llcrnrlon = -110.5
    llcrnrlat = 35.0
    urcrnrlon = -100.5
    urcrnrlat = 42.0
    lat_0 = 38.0
    lon_0 = -105.0
    xscale=0.17
    yscale=0.18
  elif dom == 'LA':
    llcrnrlon = -121.0
    llcrnrlat = 32.0
    urcrnrlon = -114.0
    urcrnrlat = 37.0
    lat_0 = 34.0
    lon_0 = -114.0
    xscale=0.16
    yscale=0.18
  elif dom == 'MA':
    llcrnrlon = -82.0
    llcrnrlat = 36.5
    urcrnrlon = -73.5
    urcrnrlat = 42.0
    lat_0 = 37.5
    lon_0 = -80.0
    xscale=0.18
    yscale=0.18
  elif dom == 'NC':
    llcrnrlon = -111.0
    llcrnrlat = 39.0
    urcrnrlon = -93.5
    urcrnrlat = 49.0
    lat_0 = 44.5
    lon_0 = -102.0
    xscale=0.16
    yscale=0.18
  elif dom == 'NE':
    llcrnrlon = -80.0     
    llcrnrlat = 40.5
    urcrnrlon = -66.0
    urcrnrlat = 47.5
    lat_0 = 42.0
    lon_0 = -80.0
    xscale=0.16
    yscale=0.18
  elif dom == 'NW':
    llcrnrlon = -125.5     
    llcrnrlat = 40.5
    urcrnrlon = -109.0
    urcrnrlat = 49.5
    lat_0 = 44.0
    lon_0 = -116.0
    xscale=0.15
    yscale=0.18
  elif dom == 'OV':
    llcrnrlon = -91.5 
    llcrnrlat = 34.75
    urcrnrlon = -80.0
    urcrnrlat = 43.0
    lat_0 = 38.0
    lon_0 = -87.0          
    xscale=0.18
    yscale=0.17
  elif dom == 'SC':
    llcrnrlon = -108.0 
    llcrnrlat = 25.0
    urcrnrlon = -88.0
    urcrnrlat = 37.0
    lat_0 = 32.0
    lon_0 = -98.0      
    xscale=0.14
    yscale=0.18
  elif dom == 'SE':
    llcrnrlon = -91.5 
    llcrnrlat = 24.0
    urcrnrlon = -74.0
    urcrnrlat = 36.5
    lat_0 = 34.0
    lon_0 = -85.0
    xscale=0.17
    yscale=0.18
  elif dom == 'SF':
    llcrnrlon = -123.25 
    llcrnrlat = 37.25
    urcrnrlon = -121.25
    urcrnrlat = 38.5
    lat_0 = 37.5
    lon_0 = -121.0
    xscale=0.16
    yscale=0.19
  elif dom == 'SP':
    llcrnrlon = -125.0
    llcrnrlat = 45.0
    urcrnrlon = -119.5
    urcrnrlat = 49.2
    lat_0 = 46.0
    lon_0 = -120.0
    xscale=0.19
    yscale=0.18
  elif dom == 'SW':
    llcrnrlon = -125.0 
    llcrnrlat = 30.0
    urcrnrlon = -108.0
    urcrnrlat = 42.5
    lat_0 = 37.0
    lon_0 = -113.0
    xscale=0.17
    yscale=0.18
  elif dom == 'UM':
    llcrnrlon = -96.75 
    llcrnrlat = 39.75
    urcrnrlon = -81.0
    urcrnrlat = 49.0
    lat_0 = 44.0
    lon_0 = -91.5
    xscale=0.18
    yscale=0.18

  # Create basemap instance and set the dimensions
  for ax in axes:
    if dom == 'BN' or dom == 'LA' or dom == 'SF' or dom == 'SP':
      m = Basemap(ax=ax,projection='gnom',lat_0=lat_0,lon_0=lon_0,\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                  resolution='h')
    else:
      m = Basemap(ax=ax,projection='gnom',lat_0=lat_0,lon_0=lon_0,\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                  resolution='l')
    m.fillcontinents(color='LightGrey',zorder=0)
    m.drawcoastlines(linewidth=0.75)
    m.drawstates(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
##  parallels = np.arange(0.,90.,10.)
##  map.drawparallels(parallels,labels=[1,0,0,0],fontsize=6)
##  meridians = np.arange(180.,360.,10.)
##  map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=6)
    x,y   = m(lon,lat)
    x2,y2 = m(lon2,lat2)
  
    x_shift,y_shift   = m(lon_shift,lat_shift)
    x2_shift,y2_shift = m(lon2_shift,lat2_shift)
  
  # Map/figure has been set up here, save axes instances for use again later
    if par == 1:
      keep_ax_lst_1 = ax.get_children()[:]
    elif par == 2:
      keep_ax_lst_2 = ax.get_children()[:]
    elif par == 3:
      keep_ax_lst_3 = ax.get_children()[:]
    elif par == 4:
      keep_ax_lst_4 = ax.get_children()[:]

    par += 1
  par = 1


  clevsdif = [-0.12,-0.10,-0.08,-0.06,-0.04,-0.02,0.,0.02,0.04,0.06,0.08,0.10,0.12]
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)
  skip = 30

################################
  # Plot HGT
################################
  t1 = time.clock()
  print('Working on hgt for '+dom)

  units = 'm'

  if level == 200 or level == 250 or level == 300:
      cint = np.arange(720,1200,12)
  elif level == 500:
      cint = np.arange(498.,651.,6.)
  elif level == 500:
      cint = np.arange(265.,366.,5.)
  elif level == 850:
      cint = np.arange(120.,181.,3.)
  elif level == 1000:
      cint = np.arange(0.,181.,3.)

  contour_var_1 = hgt_list_1[levels.index(level)] * 0.1
  contour_var_2 = hgt_list_2[levels.index(level)] * 0.1
  hgt_dif = hgt_list_2[levels.index(level)] - hgt_list_1[levels.index(level)]

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      contours = m.contour(x,y,contour_var_1,cint,colors='k',linewidths=1.5,ax=ax)
      ax.text(.5,1.03,prod_str+' '+str(level)+' mb Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))

    elif par == 2:
      contours = m.contour(x,y,contour_var_2,cint,colors='k',linewidths=1.5,ax=ax)
      ax.text(.5,1.03,para_str+' '+str(level)+' mb Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,hgt_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label('Diff Sum = '+str(np.sum(hgt_dif))+' '+units+'\n Avg Diff = '+str(np.sum(hgt_dif)/(nx*ny))+' '+units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,para_str+' - '+prod_str+' '+str(level)+' mb Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))


    par += 1
  par = 1

  compress_and_save('compare_hgt'+str(level)+'_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot hgt for: '+dom) % t3)

#################################
  # Plot TMP
#################################
  t1 = time.clock()
  print('Working on tmp for '+dom)

  # Clear off old plottables but keep all the map info
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = u'\xb0''C'


  if level == 200 or level == 250 or level == 300:
      clevs = np.linspace(-70,-30,41)
  elif level == 500:
      clevs = np.linspace(-30,10,41)
  elif level == 850:
      clevs = np.linspace(-30,50,81)
  elif level == 1000:
      clevs = np.linspace(0,50,51)


  cm = cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  fill_var_1 = tmp_list_1[levels.index(level)]
  fill_var_2 = tmp_list_2[levels.index(level)]
  tmp_dif = tmp_list_2[levels.index(level)] - tmp_list_1[levels.index(level)]

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,fill_var_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,prod_str+' '+str(level)+' mb Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
 
    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,fill_var_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,para_str+' '+str(level)+' mb Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,tmp_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
    # cbar3.set_label(units,fontsize=6)
      cbar3.set_label('Diff Sum = '+str(np.sum(tmp_dif))+' '+units+'\n Avg Diff = '+str(np.sum(tmp_dif)/(nx*ny))+' '+units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,para_str+' - '+prod_str+' '+str(level)+' mb Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2')) 

    par += 1
  par = 1

  compress_and_save('compare_tmp'+str(level)+'_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot tmp for: '+dom) % t3)



#################################
  # Plot DPT
#################################
  t1 = time.clock()
  print('Working on dpt for '+dom)

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = u'\xb0''C'


  if level == 200 or level == 250 or level == 300:
      clevs = np.linspace(-100,-20,81)
  elif level == 500:
      clevs = np.linspace(-70,20,91)
  elif level == 850:
      clevs = np.linspace(-30,50,81)
  elif level == 1000:
      clevs = np.linspace(-30,50,81)


  cm = cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  fill_var_1 = dpt_list_1[levels.index(level)]
  fill_var_2 = dpt_list_2[levels.index(level)]
  dpt_dif = dpt_list_2[levels.index(level)] - dpt_list_1[levels.index(level)]

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,fill_var_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,prod_str+' '+str(level)+' mb DPT ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
 
    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,fill_var_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,para_str+' '+str(level)+' mb DPT ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,dpt_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
    # cbar3.set_label(units,fontsize=6)
      cbar3.set_label('Diff Sum = '+str(np.sum(dpt_dif))+' '+units+'\n Avg Diff = '+str(np.sum(dpt_dif)/(nx*ny))+' '+units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,para_str+' - '+prod_str+' '+str(level)+' mb DPT ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2')) 

    par += 1
  par = 1

  compress_and_save('compare_dpt'+str(level)+'_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot dpt for: '+dom) % t3)


#################################
  # Plot VVEL
#################################
  t1 = time.clock()
  print('Working on vvel for '+dom)

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'Pa/s'


  if level == 200 or level == 250 or level == 300:
      clevs = np.arange(-10.,11.,2.)
  elif level == 500:
      clevs = np.arange(-20.,21.,2.)
  elif level == 850:
      clevs = np.arange(-20.,21.,2.)
  elif level == 1000:
      clevs = np.arange(-20.,21.,2.)




  cm = cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  fill_var_1 = vvel_list_1[levels.index(level)]
  fill_var_2 = vvel_list_2[levels.index(level)]
  vvel_dif = vvel_list_2[levels.index(level)] - vvel_list_1[levels.index(level)]

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,fill_var_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,prod_str+' '+str(level)+' mb VVEL ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))

    elif par == 2:
      cs_1 = m.pcolormesh(x_shift,y_shift,fill_var_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,para_str+' '+str(level)+' mb VVEL ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,vvel_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
    # cbar3.set_label(units,fontsize=6)
      cbar3.set_label('Diff Sum = '+str(np.sum(vvel_dif))+' '+units+'\n Avg Diff = '+str(np.sum(vvel_dif)/(nx*ny))+' '+units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,para_str+' - '+prod_str+' '+str(level)+' mb VVEL ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))


    par += 1
  par = 1

  compress_and_save('compare_vvel'+str(level)+'_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot vvel for: '+dom) % t3)


  plt.clf()

#################################
  # Plot WIND
#################################
  t1 = time.clock()
  print('Working on ugrd/vgrd for '+dom)

  # create figure and axes instances
  fig = plt.figure()
  gs = GridSpec(9,9,wspace=0.0,hspace=0.0)
  ax1 = fig.add_subplot(gs[0:4,0:4])
  ax2 = fig.add_subplot(gs[0:4,5:])
  ax3 = fig.add_subplot(gs[5:,0:4])
  ax4 = fig.add_subplot(gs[5:,5:])
  axes = [ax1, ax2, ax3, ax4]
  par = 1

  # Map corners for each domain
  if dom == 'conus':
    llcrnrlon = -125.5
    llcrnrlat = 18.0
    urcrnrlon = -57.5
    urcrnrlat = 52.0
    lat_0 = 35.4
    lon_0 = -97.6
    xscale=0.15
    yscale=0.2

  # Create basemap instance and set the dimensions
  for ax in axes:
    if dom == 'BN' or dom == 'LA' or dom == 'SF' or dom == 'SP':
      m = Basemap(ax=ax,projection='gnom',lat_0=lat_0,lon_0=lon_0,\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                  resolution='h')
    else:
      m = Basemap(ax=ax,projection='gnom',lat_0=lat_0,lon_0=lon_0,\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                  resolution='l')
    m.fillcontinents(color='LightGrey',zorder=0)
    m.drawcoastlines(linewidth=0.75)
    m.drawstates(linewidth=0.5)
    m.drawcountries(linewidth=0.5)

    x,y   = m(lon,lat)
    x2,y2 = m(lon2,lat2)
  
    x_shift,y_shift   = m(lon_shift,lat_shift)
    x2_shift,y2_shift = m(lon2_shift,lat2_shift)


  if level == 200 or level == 250 or level == 300:
      clevs = np.arange(30.,151.,2.5)
  elif level == 500:
      clevs = np.arange(10.,91.,2.5)
  elif level == 850:
      clevs = np.arange(5.,81.,2.5)
  elif level == 1000:
      clevs = np.arange(5.,81.,2.5)


  units = 'kts'

  cm = cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  fill_var_1 = np.sqrt((ugrd_list_1[levels.index(level)]*1.94384)**2 + (vgrd_list_1[levels.index(level)]*1.94384)**2)
  fill_var_2 = np.sqrt((ugrd_list_2[levels.index(level)]*1.94384)**2 + (vgrd_list_2[levels.index(level)]*1.94384)**2)
  ugrd_dif = ugrd_list_2[levels.index(level)] - ugrd_list_1[levels.index(level)]
  vgrd_dif = vgrd_list_2[levels.index(level)] - vgrd_list_1[levels.index(level)]

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,fill_var_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,prod_str+' '+str(level)+' mb Isotachs ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,fill_var_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,para_str+' '+str(level)+' mb Isotachs ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,ugrd_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
     #cbar3.set_label(units,fontsize=6)
      cbar3.set_label('Diff Sum = '+str(np.sum(ugrd_dif))+' '+units+'\n Avg Diff = '+str(np.sum(ugrd_dif)/(nx*ny))+' '+units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,para_str+' - '+prod_str+' '+str(level)+' mb UGRD ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))

    elif par == 4:
      cs = m.pcolormesh(x2_shift,y2_shift,vgrd_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
    # cbar3.set_label(units,fontsize=6)
      cbar3.set_label('Diff Sum = '+str(np.sum(vgrd_dif))+' '+units+'\n Avg Diff = '+str(np.sum(vgrd_dif)/(nx*ny))+' '+units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,para_str+' - '+prod_str+' '+str(level)+' mb VGRD ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))


    par += 1
  par = 1

  compress_and_save('compare_wind'+str(level)+'_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot wind for: '+dom) % t3)


  plt.clf()


######################################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all variables for: "+dom) % t3dom)
  plt.clf()

######################################################



######################################################

#for domain in domains:
#    plot_all(domain)

main()
#plot_all('conus')
