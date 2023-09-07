#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
#from matplotlib import GridSpec, rcParams, colors
from matplotlib import colors as c
from pylab import *
import numpy as np
import pygrib, datetime, time, os, sys, subprocess
import ncepy, scipy
from ncepgrib2 import Grib2Encode, Grib2Decode
import dawsonpy
import itertools

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


# Get machine and head directory
machine, hostname = dawsonpy.get_machine()

case = 'AMS'
machine = 'WCOSS_DELL_P3'
if machine == 'WCOSS':
    DIR = '/gpfs/'+hostname[0]+'p2/ptmp/'+os.environ['USER']+'/rap_hrrr_retros'
    DATA_DIR = os.path.join('/ptmpp2/Logan.Dawson/MEG/', case, 'data')
    GRAPHX_DIR = os.path.join('/ptmpp2/Logan.Dawson/MEG/', case, 'graphx')
elif machine == 'WCOSS_DELL_P3':
    DATA_DIR = os.path.join('/gpfs/dell2/ptmp/Logan.Dawson/MEG/', case, 'data')
    GRAPHX_DIR = os.path.join('/gpfs/dell2/ptmp/Logan.Dawson/MEG/', case, 'graphx')

# Create graphx directory
if os.path.exists(DATA_DIR):
   if not os.path.exists(GRAPHX_DIR):
      os.makedirs(GRAPHX_DIR)
else:
   raise NameError, 'data for '+case+' case not found'


# Determine initial date/time
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
print YYYY, MM, DD, HH

date_str = datetime.datetime(YYYY,MM,DD,HH,0,0)


large_domains = ['CONUS','Australia']
regional_domains = ['eCONUS']
subregional_domains = ['SPL','MIDSOUTH','Barry']
state_domains = ['OK','Louisiana']

ARWs = ['HIRESWARW','HIRESWARW2','HRRR','RAP']
NMMBs = ['NAM3','HIRESWNMMB']


#Specify plots to make
domains = ['SPL']

fields = ['refc_uh','t2_10mwind','apcp24']

plots = [n for n in itertools.product(domains,fields)]
print plots

fhours = [36]
valid_list = [date_str + datetime.timedelta(hours=x) for x in fhours]

fhrs = np.arange(12,37,3)

domain = 'SPL'
field = fields[2]

for j in range(len(valid_list)):
#for j in range(1):
   print fhours[j]
   print valid_list[j]
   YYYYMMDDCC = valid_list[j].strftime("%Y%m%d%H")

   fig = plt.figure(figsize=(13,7))
   ax1 = fig.add_subplot(231)
   ax2 = fig.add_subplot(232)
   ax3 = fig.add_subplot(233)
   ax4 = fig.add_subplot(234)
   ax5 = fig.add_subplot(235)
   ax6 = fig.add_subplot(236)
   axes = [ax1,ax2,ax3,ax4,ax5,ax6]

   k = 0
   for ax in axes:

      print 'plotting '+str.lower(field)+' at F'+str(fhours[j])+' on '+domain+' domain'
    # domain = domains[0]  # OK
      m = Basemap(llcrnrlon=-107.5,llcrnrlat=30.,urcrnrlon=-90.,urcrnrlat=40.,\
                  resolution='i',projection='lcc',\
                  lat_1=25.,lat_2=46.,lon_0=-97.5,area_thresh=1000.,ax=ax)

      m.drawcoastlines()
      m.drawstates(linewidth=0.75)
      m.drawcountries()

      if domain in large_domains:
         latlongrid = 10.
         barb_length = 4.5
         parallels = np.arange(-90.,91.,latlongrid)
         m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
         meridians = np.arange(0.,360.,latlongrid)
         m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
      else:
         m.drawcounties()
         latlongrid = 5.
         barb_length = 5.5


      APCP = []
      if k == 0:
         model_str = 'HRRR'

         for fhr in fhrs:
             fil = DATA_DIR+'/'+str.lower(model_str)+'.'+cycle[0:8]+'.t'+ \
                   cycle[8:10]+'z.wrfprsf'+str(fhr).zfill(2)+'.grib2'

             grbs = pygrib.open(fil)
             grb = grbs.select(name='Total Precipitation')[0]
             rt_bucket = grb.values/25.4
             APCP.append(rt_bucket)

      elif k == 1:
         model_str = 'HiResWARW'

         for fhr in fhrs:
             fil = DATA_DIR+'/'+str.lower(model_str[0:6])+'.'+cycle[0:8]+'.t'+ \
                   cycle[8:10]+'z.'+str.lower(model_str[6:])+'_5km.f'+str(fhr).zfill(2)+'.conus.grib2'

             grbs = pygrib.open(fil)
             grb = grbs.select(name='Total Precipitation')[0]
             rt_bucket = grb.values/25.4
             APCP.append(rt_bucket)

      elif k == 2:
         model_str = 'HiResWARW2'

         for fhr in fhrs:
             fil = DATA_DIR+'/'+str.lower(model_str[0:6])+'.'+cycle[0:8]+'.t'+ \
                   cycle[8:10]+'z.'+str.lower(model_str[6:9])+'_5km.f'+str(fhr).zfill(2)+'.conusmem2.grib2'

             grbs = pygrib.open(fil)
             grb = grbs.select(name='Total Precipitation')[0]
             rt_bucket = grb.values/25.4
             APCP.append(rt_bucket)

      elif k == 3:
         model_str = 'NAM3'

         for fhr in fhrs:
             fil = DATA_DIR+'/'+str.lower(model_str[0:3])+'.'+cycle[0:8]+'.t'+ \
                   cycle[8:10]+'z.conusnest.hiresf'+str(fhr).zfill(2)+'.tm00.grib2'

             grbs = pygrib.open(fil)
             grb = grbs.select(name='Total Precipitation')[0]
             bucket = grb.values/25.4
             APCP.append(bucket)

      elif k == 4:
         model_str = 'HiResWNMMB'

         for fhr in fhrs:
             fil = DATA_DIR+'/'+str.lower(model_str[0:6])+'.'+cycle[0:8]+'.t'+ \
                   cycle[8:10]+'z.'+str.lower(model_str[6:])+'_5km.f'+str(fhr).zfill(2)+'.conus.grib2'

             grbs = pygrib.open(fil)
             grb = grbs.select(name='Total Precipitation')[0]
             bucket = grb.values/25.4
             APCP.append(bucket)

      elif k == 5 and field[0:4] == 'refc':
         model_str = 'OBS'
         fil = DATA_DIR+'/refd3d.'+YYYYMMDDCC[0:8]+'.t'+ \
               YYYYMMDDCC[8:10]+'z.grb2f00'

      elif k == 5 and field[0:2] == 't2':
         model_str = 'URMA'
         anl_str = model_str
         fil = DATA_DIR+'/'+str.lower(anl_str)+'2p5.'+YYYYMMDDCC[0:8]+'.t'+ \
               YYYYMMDDCC[8:10]+'z.2dvaranl_ndfd.grb2'

      elif k == 5 and field[0:4] == 'apcp':
         model_str = 'Stage IV'
         anl_str = 'ST4'
         fil = DATA_DIR+'/'+str.upper(anl_str)+'.'+YYYYMMDDCC+'.24h'

         grbs = pygrib.open(fil)
         grb = grbs.select(name='Total Precipitation')[0]
         bucket = grb.values/25.4
         APCP.append(bucket)


      lats, lons = grb.latlons()         



      if str.upper(model_str[0:6]) == 'HIRESW':
         HLwindow = 250
         if domain in large_domains:
            skip = 30
         elif domain in subregional_domains:
            skip = 15
         elif domain in state_domains:
            skip = 10

      elif str.upper(model_str) == 'NAM3' or str.upper(model_str) == 'HRRR':
         HLwindow = 300
         if domain in large_domains:
            skip = 50
         elif domain in subregional_domains:
            skip = 25
         elif domain in state_domains:
            skip = 20

      elif str.upper(model_str) == 'RTMA' or str.upper(model_str) == 'URMA':
         HLwindow = 400
         if domain in large_domains:
            skip = 75
         elif domain in subregional_domains:
            skip = 30
         elif domain in state_domains:
            skip = 20



      # Temperature
      if field[0:2] == 't2':
         t2  = grbs.select(stepType='instant',name='2 metre temperature')[0].values*9/5 - 459.67
         u10 = grbs.select(stepType='instant',name='10 metre U wind component')[0].values*1.94384
         v10 = grbs.select(stepType='instant',name='10 metre V wind component')[0].values*1.94384

         clevs = np.arange(-16,132,4)
         tlevs = [str(clev) for clev in clevs]

         colormap = cmap_t2m()
         norm = matplotlib.colors.BoundaryNorm(clevs, colormap.N)

         fill_var = t2
         fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,cmap=colormap,norm=norm,extend='both')
         field_title = '2-m Temperature'

         try:
            if str.lower(field[3:10]) == '10mwind':

               u_var = u10
               v_var = v10

               m.barbs(lons[::skip,::skip],lats[::skip,::skip],u_var[::skip,::skip],v_var[::skip,::skip],latlon=True,length=barb_length,sizes={'spacing':0.2},pivot='middle')

            #  field_title = '2-m Temperature and 10-m Wind'

         except IndexError:
            sys.exc_clear()



      # Simulated reflectivity
      elif field[0:3] == 'ref':
         refc = grbs.select(name='Maximum/Composite radar reflectivity')[0].values

         clevs = np.linspace(5,70,14)
         tlevs = [str(clev) for clev in clevs]
         colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green', \
                      'yellow','gold','darkorange','red','firebrick','darkred','fuchsia','darkmagenta']

         if field[0:4] == 'refc':
            fill_var = refc
            field_title = 'Composite Reflectivity'
         elif field[0:5] == 'refd1':
            fill_var = refd1
         elif field[0:5] == 'refd4':
            fill_var = refd4

         fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')


      # Accumulated precip
      elif field[1:4] == 'pcp' or field[2:5] == 'pcp':
         clevs = [0.01,0.1,0.25,0.5,0.75,1,1.5,2,2.5,3,4,5,6,8,10,12,14,16,18,20]
         tlevs = [str(clev) for clev in clevs]
         colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','cyan','slateblue', \
                      'mediumorchid','darkmagenta','darkred','red','crimson','darkorange',  \
                      'salmon','yellow','goldenrod','magenta','pink','beige','black']

         stride = fhrs[1] - fhrs[0]
         print(stride)

         accum_length = 24

         if str.upper(model_str) in ARWs:
             fill_var = APCP[-1] - APCP[0]
         elif str.upper(model_str) in NMMBs:
             fill_var = np.sum(np.array(APCP),axis=0)-APCP[0]
         elif model_str in 'Stage IV':
             fill_var = APCP[0]

         print(np.shape(APCP),np.shape(fill_var),np.shape(lats),np.shape(lons))
         fill = m.contourf(lons,lats,fill_var,clevs,latlon=True,colors=colorlist,extend='max')


      if str.upper(model_str) == 'NAM3':
          mod_str = str.upper(model_str[0:3])+' Nest'
      elif str.upper(model_str[0:6]) == 'HIRESW':
          mod_str = model_str[0:6]+' '+str.upper(model_str[6:])
      elif str.lower(model_str[4:]) == 'mean':
          mod_str = str.upper(model_str[0:4])+' Mean'
      else:
          mod_str = str.upper(model_str)
          mod_str = model_str

      ax.text(0.5, 0.05, mod_str,  horizontalalignment='center', weight='bold', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.85))

      grbs.close()
      k += 1


   cax = fig.add_axes([0.2,0.01,0.6,0.03])
   cbar = plt.colorbar(fill,cax=cax,ticks=clevs,orientation='horizontal')
   cbar.ax.set_xticklabels(tlevs)
   cbar.ax.tick_params(labelsize=10)

   for label in cbar.ax.xaxis.get_ticklabels():
      label.set_visible(False)

   if field[0:2] == 't2':
      cbar.set_label(u'\xb0''F')

      for label in cbar.ax.xaxis.get_ticklabels()[::4]:
         label.set_visible(True)

   elif field[0:3] == 'ref':
      cbar.set_label('dBZ')

      for label in cbar.ax.xaxis.get_ticklabels():
         label.set_visible(True)

   elif field[0:4] == 'apcp':
      cbar.set_label('in')

      for label in cbar.ax.xaxis.get_ticklabels():
         label.set_visible(True)



 # titlestr = '00Z 5/20 '+field_title+' Forecasts and Analysis '+valid_list[j].strftime('Valid at %HZ %d %b %Y')+' (F'+str(fhrs[j]).zfill(2)+')'
   titlestr = '00Z 5/20 24-h QPFs and Stage IV QPE '+valid_list[j].strftime('Valid at %HZ %d %b %Y')+' (F'+str(fhrs[-1]).zfill(2)+')'
   fig.suptitle(titlestr, size=18, weight='bold')

   fname = str.lower(field)+'_'+str.lower(domain)+'_'+cycle+'_f'+str(fhrs[-1]).zfill(2)

   plt.tight_layout()
   fig.subplots_adjust(top=0.96,bottom=0.04)
   plt.savefig(GRAPHX_DIR+'/'+fname+'.png',bbox_inches='tight')
   plt.close()



