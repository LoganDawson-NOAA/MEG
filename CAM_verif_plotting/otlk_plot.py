#!/usr/bin/env python
import netCDF4
import numpy as np
import os, sys, datetime, time, subprocess
import re, csv, glob
import multiprocessing, itertools, collections

import pyproj
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

import matplotlib
import matplotlib.image as image
from matplotlib.gridspec import GridSpec
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy
import dawsonpy


current_time = datetime.datetime.utcnow()

grid = str(sys.argv[1])

# Get machine
#machine, hostname = dawsonpy.get_machine()

# Set up working directory
NOSCRUB_DIR = '/lfs/h2/emc/vpppg/noscrub/'+os.environ['USER']+'/evs/v1.0/prep/cam/spc_otlk.20230405'
PTMP_DIR    = '/lfs/h2/emc/ptmp/'+os.environ['USER']+'/bukovsky'

if not os.path.exists(PTMP_DIR):
    os.makedirs(PTMP_DIR)


otlk_fil = NOSCRUB_DIR+'/spc_otlk.day1_1200_MRGL.v2023040512-2023040612.G227.nc'
#otlk_file = '/lfs/h2/emc/ptmp/logan.dawson/CAM_verif/METplus/metplus.out/masks/G227/SPC_outlooks/day1otlk_20230404_1200_cat_Rec2_SLGT_mask.nc'
nc = netCDF4.Dataset(otlk_fil,'r')
otlk = nc.variables['DAY1_1200_MRGL'][:]
grid_lats = nc.variables['lat'][:]
grid_lons = nc.variables['lon'][:]
nc.close()

fields = ['otlk']

large_domains = ['conus']
domains = ['conus']

plots = [n for n in itertools.product(domains,fields)]
print(plots)




def plot_fields(plot):

    thing = np.asarray(plot)
    domain = thing[0]
    field = thing[1]

    fig = plt.figure(figsize=(10.9,8.9))
    gs = GridSpec(1,1,wspace=0.0,hspace=0.0)

    # Define where Cartopy maps are located
    cartopy.config['data_dir'] = '/lfs/h2/emc/vpppg/save/'+os.environ['USER']+'/python/NaturalEarth'

    back_res='50m'
    back_img='off'


    if domain == 'conus':
        llcrnrlon=-121.0
        llcrnrlat=21.0
        urcrnrlon=-62.6
        urcrnrlat=49.0
        xscale=0.15
        yscale=0.2


    extent = [llcrnrlon-1,urcrnrlon-1,llcrnrlat-1,urcrnrlat+1]
    myproj = ccrs.LambertConformal(central_longitude=-101.,central_latitude=35.4,
             false_easting=0.0,false_northing=0.0,secant_latitudes=None,
             standard_parallels=(32,46),globe=None)

    ax = fig.add_subplot(gs[0:1,0:1], projection=myproj)
    ax.set_extent(extent)
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

    # All lat lons are earth relative, so setup the associated projection correct for that data
    transform = ccrs.PlateCarree()

  # high-resolution background images
    if back_img=='on':
        img = plt.imread('/lfs/h2/emc/vpppg/save/'+os.environ['USER']+'/python/NaturalEarth/raster_files/NE1_50M_SR_W.tif')
        ax.imshow(img, origin='upper', transform=transform)


    if domain in large_domains:
        latlongrid = 10.
        parallels = np.arange(-90.,91.,latlongrid)
 #      m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
        meridians = np.arange(0.,360.,latlongrid)
 #      m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    else:
        ax.add_feature(COUNTIES,facecolor='none',edgecolor='gray')

#   ax.add_feature(states)
#   ax.add_feature(borders)
#   ax.add_feature(coastlines)
    ax.add_feature(cfeature.STATES.with_scale('50m'),linewidths=0.3,linestyle='solid',edgecolor='k',zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidths=0.5,linestyle='solid',edgecolor='k',zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidths=0.6,linestyle='solid',edgecolor='k',zorder=4)


#   x, y = m(grid_lons,grid_lats)
#   x, y = np.meshgrid(grid_lons,grid_lats)
#   lon_shift = grid_lons - (dx/2.)
#   lat_shift = grid_lats - (dx/2.)
#   x_shift, y_shift = np.meshgrid(lon_shift,lat_shift)


    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))



    if field == 'otlk':

        clevs = [1,1000]
        colorlist = ['red']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        fill = ax.pcolormesh(grid_lons,grid_lats,otlk,cmap=cm,vmin=1,norm=norm,transform=transform)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')



    elif field == 'regions':


        clevs = [1,1000]
        colorlist = ['green']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        east_cov = n_east/n_conus
        fill = ax.pcolormesh(grid_lons,grid_lats,conus_east,cmap=cm,vmin=1,norm=norm,transform=transform)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')


        clevs = [1,1000]
        colorlist = ['red']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        central_cov = n_central/n_conus
        fill = ax.pcolormesh(grid_lons,grid_lats,conus_central,cmap=cm,vmin=1,norm=norm,transform=transform)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')


        clevs = [1,1000]
        colorlist = ['purple']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        west_cov = n_west/n_conus
        fill = ax.pcolormesh(grid_lons,grid_lats,conus_west,cmap=cm,vmin=1,norm=norm,transform=transform)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')


        clevs = [1,1000]
        colorlist = ['blue']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        south_cov = n_south/n_conus
        fill = ax.pcolormesh(grid_lons,grid_lats,conus_south,cmap=cm,vmin=1,norm=norm,transform=transform)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')


    outfile = field+'20230405_'+grid 

    plt.savefig(PTMP_DIR+'/'+outfile+'.png',bbox_inches='tight')
    plt.close()



def main():

   pool = multiprocessing.Pool(len(plots))
   pool.map(plot_fields,plots)



main()
