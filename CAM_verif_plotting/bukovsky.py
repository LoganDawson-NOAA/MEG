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
NOSCRUB_DIR = '/lfs/h2/emc/vpppg/noscrub/'+os.environ['USER']+'/CAM_verif/masks/Bukovsky_CONUS/EVS_fix'
PTMP_DIR    = '/lfs/h2/emc/ptmp/'+os.environ['USER']+'/bukovsky'

if not os.path.exists(PTMP_DIR):
    os.makedirs(PTMP_DIR)


conus_fil = NOSCRUB_DIR+'/Bukovsky_'+grid+'_CONUS.nc'
nc = netCDF4.Dataset(conus_fil,'r')
conus = nc.variables['CONUS'][:]
grid_lats = nc.variables['lat'][:]
grid_lons = nc.variables['lon'][:]
nc.close()

n_conus = np.sum(conus)
fields = ['conus']


if grid != 'G211':
    east_fil = NOSCRUB_DIR+'/Bukovsky_'+grid+'_CONUS_East.nc'
    nc = netCDF4.Dataset(east_fil,'r')
    conus_east = nc.variables['CONUS_East'][:]
    nc.close()
    n_east = np.sum(conus_east)

    west_fil = NOSCRUB_DIR+'/Bukovsky_'+grid+'_CONUS_West.nc'
    nc = netCDF4.Dataset(west_fil,'r')
    conus_west = nc.variables['CONUS_West'][:]
    nc.close()
    n_west = np.sum(conus_west)

    central_fil = NOSCRUB_DIR+'/Bukovsky_'+grid+'_CONUS_Central.nc'
    nc = netCDF4.Dataset(central_fil,'r')
    conus_central = nc.variables['CONUS_Central'][:]
    nc.close()
    n_central = np.sum(conus_central)

    south_fil = NOSCRUB_DIR+'/Bukovsky_'+grid+'_CONUS_South.nc'
    nc = netCDF4.Dataset(south_fil,'r')
    conus_south = nc.variables['CONUS_South'][:]
    nc.close()
    n_south = np.sum(conus_south)

    fields.extend(['regions'])


if grid == 'G104':
    pacificnw_fil = NOSCRUB_DIR+'/PacificNW.nc'
    pacificsw_fil = NOSCRUB_DIR+'/PacificSW.nc'
    southwest_fil = NOSCRUB_DIR+'/Southwest.nc'
    mezquital_fil = NOSCRUB_DIR+'/Mezquital.nc'
    nrockies_fil = NOSCRUB_DIR+'/NRockies.nc'
    srockies_fil = NOSCRUB_DIR+'/SRockies.nc'
    reatbasin_fil = NOSCRUB_DIR+'/GreatBasin.nc'
    nplains_fil = NOSCRUB_DIR+'/NPlains.nc'
    cplains_fil = NOSCRUB_DIR+'/CPlains.nc'
    splains_fil = NOSCRUB_DIR+'/SPlains.nc'
    prairie_fil = NOSCRUB_DIR+'/Prairie.nc'
    greatlakes_fil = NOSCRUB_DIR+'/GreatLakes.nc'
    appalachia_fil = NOSCRUB_DIR+'/Appalachia.nc'
    deepsouth_fil = NOSCRUB_DIR+'/DeepSouth.nc'
    southeast_fil = NOSCRUB_DIR+'/Southeast.nc'
    midatlantic_fil = NOSCRUB_DIR+'/MidAtlantic.nc'
    northeast_fil = NOSCRUB_DIR+'/NorthAtlantic.nc'
    
    nc = netCDF4.Dataset(pacificnw_fil,'r')
    pacificnw = nc.variables['PacificNW'][:]
    nc.close()
    
    nc = netCDF4.Dataset(pacificsw_fil,'r')
    pacificsw = nc.variables['PacificSW'][:]
    nc.close()
    
    nc = netCDF4.Dataset(southwest_fil,'r')
    southwest = nc.variables['Southwest'][:]
    nc.close()
    
    nc = netCDF4.Dataset(mezquital_fil,'r')
    mezquital = nc.variables['Mezquital'][:]
    nc.close()
    
    nc = netCDF4.Dataset(nrockies_fil,'r')
    nrockies = nc.variables['NRockies'][:]
    nc.close()
    
    nc = netCDF4.Dataset(srockies_fil,'r')
    srockies = nc.variables['SRockies'][:]
    nc.close()
    
    nc = netCDF4.Dataset(greatbasin_fil,'r')
    greatbasin = nc.variables['GreatBasin'][:]
    nc.close()
    
    nc = netCDF4.Dataset(nplains_fil,'r')
    nplains = nc.variables['NPlains'][:]
    nc.close()
    
    nc = netCDF4.Dataset(cplains_fil,'r')
    cplains = nc.variables['CPlains'][:]
    nc.close()
    
    nc = netCDF4.Dataset(splains_fil,'r')
    splains = nc.variables['SPlains'][:]
    nc.close()
    
    nc = netCDF4.Dataset(prairie_fil,'r')
    prairie = nc.variables['Prairie'][:]
    nc.close()
    
    nc = netCDF4.Dataset(greatlakes_fil,'r')
    greatlakes = nc.variables['GreatLakes'][:]
    nc.close()
    
    nc = netCDF4.Dataset(appalachia_fil,'r')
    appalachia = nc.variables['Appalachia'][:]
    nc.close()
    
    nc = netCDF4.Dataset(deepsouth_fil,'r')
    deepsouth = nc.variables['DeepSouth'][:]
    nc.close()
    
    nc = netCDF4.Dataset(southeast_fil,'r')
    southeast = nc.variables['Southeast'][:]
    nc.close()
    
    nc = netCDF4.Dataset(midatlantic_fil,'r')
    midatlantic = nc.variables['MidAtlantic'][:]
    nc.close()
    
    nc = netCDF4.Dataset(northeast_fil,'r')
    northeast = nc.variables['NorthAtlantic'][:]
    nc.close()

    fields.extend(['subregions'])




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
    back_img='on'


    if domain == 'conus':
        llcrnrlon=-121.0
        llcrnrlat=21.0
        urcrnrlon=-62.6
        urcrnrlat=49.0
        xscale=0.15
        yscale=0.2


    extent = [llcrnrlon-1,urcrnrlon-8,llcrnrlat-1,urcrnrlat+1]
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



    if field == 'conus':

        clevs = [1,1000]
        colorlist = ['red']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        fill = ax.pcolormesh(grid_lons,grid_lats,conus,cmap=cm,vmin=1,norm=norm,transform=transform)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')

        label_str = 'Number of CONUS grid cells: '+str(int(n_conus))


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


        label_str = 'Number of CONUS grid cells: '+str(int(n_conus))+'\n'+ \
                    'Number of West cells: '+str(int(n_west))+' ('+str(round(west_cov*100.,2))+'%)\n'+ \
                    'Number of Central cells: '+str(int(n_central))+' ('+str(round(central_cov*100.,2))+'%)\n'+ \
                    'Number of East cells: '+str(int(n_east))+' ('+str(round(east_cov*100.,2))+'%)\n'+ \
                    'Number of South cells: '+str(int(n_south))+' ('+str(round(south_cov*100.,2))+'%)'
                     

    ax.text(0.05, 0.04, label_str,  horizontalalignment='left', weight='bold', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.99,boxstyle='square,pad=0.2'))

    outfile = field+'_'+grid 

    plt.savefig(PTMP_DIR+'/'+outfile+'.png',bbox_inches='tight')
    plt.close()



def main():

   pool = multiprocessing.Pool(len(plots))
   pool.map(plot_fields,plots)



main()
