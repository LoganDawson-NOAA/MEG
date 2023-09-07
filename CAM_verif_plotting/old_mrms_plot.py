#!/usr/bin/env python
import netCDF4
import numpy as np
import os, sys, datetime, time, subprocess
import re, csv, glob
import multiprocessing, itertools, collections
import ncepy, dawsonpy

import pyproj
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cartopy

import matplotlib
import matplotlib.image as image
from matplotlib.gridspec import GridSpec
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as c

vtime = str(sys.argv[1])

# Get machine
#machine, hostname = dawsonpy.get_machine()

# Set up working directory
DATA_DIR = '/lfs/h2/emc/ptmp/'+os.environ['USER']+'/CAM_verif/METplus/metplus.out/regrid/G227/MRMS/'+vtime[0:8]
GRAPHX_DIR = '/lfs/h2/emc/ptmp/'+os.environ['USER']+'/MRMS_graphx/'+vtime[0:8]

if not os.path.exists(GRAPHX_DIR):
    os.makedirs(GRAPHX_DIR)


YYYY = int(vtime[0:4])
MM   = int(vtime[4:6])
DD   = int(vtime[6:8])
HH   = int(vtime[8:10])

valid_time = datetime.datetime(YYYY,MM,DD,HH,00,00)
valid_str  = valid_time.strftime('Valid: %HZ %d %b %Y')

# Python output
#refc_fil = DATA_DIR+'/MergedReflectivityQCComposite_NBMAX_'+vtime[0:8]+'-'+vtime[8:10]+'0000_g227.nc' 
#refd_fil = DATA_DIR+'/SeamlessHSR_NBMAX_'+vtime[0:8]+'-'+vtime[8:10]+'0000_g227.nc' 
#etop_fil = DATA_DIR+'/EchoTop18_NBMAX_'+vtime[0:8]+'-'+vtime[8:10]+'0000_g227.nc' 
refc_fil = DATA_DIR+'/MergedReflectivityQCComposite_00.50_'+vtime[0:8]+'-'+vtime[8:10]+'0000_g227.nc' 
refd_fil = DATA_DIR+'/SeamlessHSR_00.00_'+vtime[0:8]+'-'+vtime[8:10]+'0000_g227.nc' 
etop_fil = DATA_DIR+'/EchoTop18_00.50_'+vtime[0:8]+'-'+vtime[8:10]+'0000_g227.nc' 

# REFC analyses
nc = netCDF4.Dataset(refc_fil,'r')
grid_lats = nc.variables['lat'][:]
grid_lons = nc.variables['lon'][:]
refc = nc.variables['MergedReflectivityQCComposite'][:]
#refcprob50 = nc.variables['MergedReflectivityQCComposite_Prob50'][:]*100.
nc.close()

# REFD analyses
nc = netCDF4.Dataset(refd_fil,'r')
refd = nc.variables['SeamlessHSR'][:]
#refdprob40 = nc.variables['SeamlessHSR_Prob40'][:]*100.
nc.close()

# RETOP analyses
nc = netCDF4.Dataset(etop_fil,'r')
retop = nc.variables['EchoTop18'][:]*3280.84*0.001
#retop = nc.variables['EchoTop18'][:]                     # my python code already converted data to kft
#retopprob30 = nc.variables['EchoTop18_Prob30'][:]*100.
nc.close()

large_domains = ['conus']

domains = ['conus','ne','se','midatl','glakes','nc','sc','nw','sw','ca']
fields = ['refc','refcprob50','refd','refdprob40','retop','retopprob30']
fields = ['refc','refd','retop']

plots = [n for n in itertools.product(domains,fields)]
print(plots)



def main():

   pool = multiprocessing.Pool(len(plots))
   pool.map(plot_fields,plots)



def plot_fields(plot):

    thing = np.asarray(plot)
    domain = thing[0]
    field = thing[1]


    print('plotting MRMS '+str.upper(field)+' on '+str.upper(domain)+' domain')

    fig = plt.figure(figsize=(10.9,8.9))
    gs = GridSpec(1,1,wspace=0.0,hspace=0.0)

    # Define where Cartopy maps are located
    cartopy.config['data_dir'] = '/lfs/h2/emc/vpppg/save/'+os.environ['USER']+'/python/NaturalEarth'

    back_res='50m'
    back_img='off'



    if domain == 'conus':
        print('specify new corner vals for CONUS')
        llcrnrlon=-121.0
        llcrnrlat=21.0
        urcrnrlon=-62.6
        urcrnrlat=49.0
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


    lon0 = -101.
    lat0 = 39.
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


#   m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,\
#               resolution='i',projection='lcc',\
#               lat_1=32.,lat_2=46.,lon_0=-101.,area_thresh=1000.,ax=ax)

#   m.drawcoastlines()
#   m.drawstates(linewidth=0.75)
#   m.drawcountries()

#   if domain in large_domains:
#       latlongrid = 10.
#       parallels = np.arange(-90.,91.,latlongrid)
#       m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
#       meridians = np.arange(0.,360.,latlongrid)
#       m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#   else:
#       m.drawcounties(linewidth=0.2, color='k')
#       latlongrid = 5.




    # REFC and REFD Obs
    if field == 'refc' or field =='refd':

        clevs = np.linspace(5,70,14)
       #colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green', \
       #             'yellow','gold','darkorange','red','firebrick','darkred','fuchsia','darkmagenta']
        colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green', \
                     '#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia','black']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

       #clevs = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]

        if field == 'refc':
            var_str = 'Composite Reflectivity'
            fname = 'refc'
            fill_var = refc
        elif field == 'refd':
            var_str = 'Seamless Hybrid Scan Reflectivity'
            fname = 'refd'
            fill_var = refd


      # fill = m.contourf(grid_lons,grid_lats,refc,clevs,cmap=ncepy.mrms_radarmap(),latlon=True,extend='max')
        fill = ax.contourf(grid_lons,grid_lats,fill_var,clevs,colors=colorlist,transform=transform,extend='max')
      # fill.set_clim(5,75)
        cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('dBZ')


    # RETOP Obs
    elif field == 'retop':

        colorlist=['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#cc66ff','#6600cc','#ff7f00','#b35900','#ff0000']
        clevs = np.arange(5.,55.,5.)
        cm = c.ListedColormap(colorlist)
        norm = c.BoundaryNorm(clevs, cm.N)

        fill_var = retop
        var_str = 'Echo Top Height'
        fname = 'retop'

   #    fill = m.contourf(grid_lons,grid_lats,retop,clevs,cmap=cm,norm=norm,latlon=True,extend='max')
        fill = ax.contourf(grid_lons,grid_lats,fill_var,clevs,cmap=cm,norm=norm,transform=transform,extend='max')
        fill.set_clim(5.,50.)
        cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('kft')


    # Probability Fields
    elif 'prob' in field:
        #clevs = [5,10,20,30,40,50,60,70,80,90,95]
        #colorlist = ['blue','dodgerblue','cyan','limegreen','chartreuse','yellow', \
        #             'orange','red','darkred','purple','orchid']

        gemlist=ncepy.gem_color_list()
        pcplist=[21,22,23,24,25,26,27,28,29,14,15]
        #Extract these colors to a new list for plotting
        pcolors=[gemlist[i] for i in pcplist]

        clevs = [5.,10.,20.,30.,40.,50.,60.,70.,80.,90.,100.]
        cm = c.ListedColormap(pcolors)
        cm.set_over('white')
        norm = c.BoundaryNorm(clevs, cm.N)

        if field == 'refcprob50':
            field_str = 'Comp. Refl. > 50 dBZ'
            fname = 'refcprob50'
            fill_var = refcprob50
        elif field == 'refdprob40':
            field_str = 'Seamless HSR > 40 dBZ'
            fname = 'refdprob40'
            fill_var = refdprob40
        elif field == 'retopprob30':
            field_str = 'Echo Top > 30 kft'
            fname = 'retopprob30'
            fill_var = retopprob30

        var_str = 'Prob. of '+field_str
      # fill = m.contourf(grid_lons,grid_lats,fill_var,clevs,cmap=cm,norm=norm,latlon=True)
        fill = ax.contourf(grid_lons,grid_lats,fill_var,clevs,cmap=cm,norm=norm,transform=transform,extend='max')
        fill.set_clim(1.0,100.)
        cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('%')


    anl_str = 'MRMS'
    plt.text(0, 1.06, anl_str, horizontalalignment='left', transform=ax.transAxes)
    plt.text(0, 1.01, var_str, horizontalalignment='left', transform=ax.transAxes)
    plt.text(1, 1.01, valid_str, horizontalalignment='right', transform=ax.transAxes)

    plt.savefig(GRAPHX_DIR+'/'+fname+'_'+domain+'_'+vtime+'.png',bbox_inches='tight')



main()


