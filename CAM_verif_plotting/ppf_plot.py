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

report_date = str(sys.argv[1])

# Get machine
#machine, hostname = dawsonpy.get_machine()

# Set up working directory
NOSCRUB_DIR = '/lfs/h2/emc/vpppg/noscrub/'+os.environ['USER']+'/CAM_verif/surrogate_severe/point2grid/'+report_date[0:6]
PTMP_DIR    = '/lfs/h2/emc/ptmp/'+os.environ['USER']+'/spc_reports/plots'

if not os.path.exists(PTMP_DIR):
    os.makedirs(PTMP_DIR)


YYYY = int(report_date[0:4])
MM   = int(report_date[4:6])
DD   = int(report_date[6:8])

valid_beg = datetime.datetime(YYYY,MM,DD,12,00,00)
valid_end = valid_beg + datetime.timedelta(hours=24)


ppf_fil = NOSCRUB_DIR+'/PracticallyPerfect_'+valid_beg.strftime('%Y%m%d%H')+'-'+valid_end.strftime('%Y%m%d%H')+'_G211.nc'
nc = netCDF4.Dataset(ppf_fil,'r')
grid_lats = nc.variables['lat'][:]
grid_lons = nc.variables['lon'][:]
ppf_data = nc.variables['LSR_PPF'][:]*100.
data = nc.variables['LSR_PPF']
nc.close()

lsr_fil = NOSCRUB_DIR+'/LocalStormReports_'+valid_beg.strftime('%Y%m%d%H')+'-'+valid_end.strftime('%Y%m%d%H')+'_G211.nc'
nc = netCDF4.Dataset(lsr_fil,'r')
lsr_data = nc.variables['Fscale_mask'][:]
nc.close()




dx = 81.271
n_osr = np.sum(lsr_data)
osr_cov = round(n_osr*81.271**2,0)

ppf_cov_step1 = np.sum(x > 2 for x in ppf_data)
ppf_cov = round(np.sum(ppf_cov_step1)*81.271**2,0)
ppf_max = round(np.max(ppf_data),2)

print('Number of OSR81s: '+str(n_osr))
print('OSR81 coverage: '+str(osr_cov)+' km^2')
print('PPF min/max: '+str(np.min(ppf_data))+'%, '+str(np.max(ppf_data))+'%')
print('PPF coverage: '+str(ppf_cov)+' km^2')


time_diff = (current_time - valid_beg).total_seconds()/3600.
lag = 8*24
if time_diff < lag:
    prelim = True
    prelim_str = '*Preliminary*'
else:
    prelim = False


fields = ['lsr','ppf']
large_domains = ['conus']
domains = ['conus']

if np.max(lsr_data) == 1:
    max_ind = np.ma.MaskedArray.argmax(ppf_data)
    ppf_max_ind = np.unravel_index(max_ind,ppf_data.shape)
    ppf_max_lat = grid_lats[ppf_max_ind]
    ppf_max_lon = grid_lons[ppf_max_ind]

    domains.extend(['dailyzoom'])

plots = [n for n in itertools.product(domains,fields)]
print(plots)



reader = shpreader.Reader('/lfs/h2/emc/lam/noscrub/Matthew.Pyle/python_tools/shp/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())


def plot_fields(plot):

    thing = np.asarray(plot)
    domain = thing[0]
    field = thing[1]


    fig = plt.figure(figsize=(10.9,8.9))
    gs = GridSpec(1,1,wspace=0.0,hspace=0.0)
#   ax = fig.add_axes([0.1,0.1,0.8,0.8])
    im = image.imread('/lfs/h2/emc/vpppg/save/logan.dawson/noaa.png')

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
        xextent=-2200000
        yextent=-885000
    elif domain == 'dailyzoom':
        llcrnrlon=ppf_max_lon-12
        llcrnrlat=ppf_max_lat-6
        urcrnrlon=ppf_max_lon+12
        urcrnrlat=ppf_max_lat+6
        print(llcrnrlat,llcrnrlon)
        print(urcrnrlat,urcrnrlon)
        xscale=0.17
        yscale=0.2
        xextent=-2200000
        yextent=-885000

    extent = [llcrnrlon-1,urcrnrlon-8,llcrnrlat-1,urcrnrlat+1]
    myproj = ccrs.LambertConformal(central_longitude=-101.,central_latitude=35.4,
             false_easting=0.0,false_northing=0.0,secant_latitudes=None,
             standard_parallels=(32,46),globe=None)

    ax = fig.add_subplot(gs[0:1,0:1], projection=myproj)
#   ax = plt.axes(projection=myproj)
    ax.set_extent(extent)
    axes = [ax]
#   ax.stock_img()

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

    if domain in large_domains:
        latlongrid = 10.
        parallels = np.arange(-90.,91.,latlongrid)
 #      m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
        meridians = np.arange(0.,360.,latlongrid)
 #      m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    else:
        ax.add_feature(COUNTIES,facecolor='none',edgecolor='gray')

    ax.add_feature(states)
    ax.add_feature(borders)
    ax.add_feature(coastlines)


#   x, y = m(grid_lons,grid_lats)
#   x_shift = x - (dx/2.)
#   y_shift = y - (dx/2.)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))


    if field == 'ppf':

        clevs = [2,5,10,15,30,45,60,70,80,90,95]
        colorlist = ['blue','dodgerblue','cyan','limegreen','chartreuse','yellow', \
                     'orange','red','darkred','purple']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

      # fill = ax.pcolormesh(x_shift,y_shift,ppf_data,cmap=cm,vmin=2,norm=norm,ax=ax)
      # fill.cmap.set_under('white',alpha=0.)
      # fill.cmap.set_over('orchid')
      # cbar = ax.colorbar(fill,ax=ax,ticks=clevs,location='bottom',pad=0.05,shrink=0.75,aspect=20,extend='max')

        fill = ax.contourf(grid_lons,grid_lats,ppf_data,clevs,transform=transform,cmap=cm,norm=norm,extend='max')
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
     #  cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20,)
        cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.67,aspect=20)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('%')

        label_str = 'Practically Perfect Max = '+str(ppf_max)+'%\n2% Coverage: '+str(ppf_cov)+' km^2'

        plt.text(0, 1.06, 'Practically Perfect \"Forecast\"', fontweight='bold', horizontalalignment='left', transform=ax.transAxes)
        plt.text(0, 1.01, '(based on SPC LSRs)', fontweight='bold', horizontalalignment='left', transform=ax.transAxes)


    elif field == 'lsr':

        clevs = [1,1000]
        colorlist = ['black']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        fill = ax.pcolormesh(grid_lons,grid_lats,lsr_data,transform=transform,cmap=cm,vmin=1,norm=norm)
#       fill = ax.pcolormesh(lon_shift,lat_shift,lsr_data,transform=transform,cmap=cm,vmin=1,norm=norm)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
  #     cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
  #     cbar.ax.tick_params(labelsize=10)
  #     cbar.set_label('%')
  #     cbar.remove()


        dummy_levs = [500,1000]
        dummy_list = ['white']
        dummy_cm = matplotlib.colors.ListedColormap(dummy_list)
        dummy_norm = matplotlib.colors.BoundaryNorm(dummy_levs, dummy_cm.N)

        dummy = ax.contourf(grid_lons,grid_lats,ppf_data,dummy_levs,transform=transform,cmap=dummy_cm,extend='max')
        dummy.cmap.set_over('white')

        cbar = plt.colorbar(dummy,ax=ax,ticks=dummy_levs,orientation='horizontal',pad=0.04,shrink=0.67,aspect=20,extend='max',drawedges=False)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('%',color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.xaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')

        label_str = 'Number of OSR81s: '+str(n_osr)+'\nCoverage: '+str(osr_cov)+' km^2'

        plt.text(0, 1.06, 'SPC Local Storm Reports', fontweight='bold', horizontalalignment='left', transform=ax.transAxes)
        plt.text(0, 1.01, '(regridded to 81-km NCEP Grid 211)', fontweight='bold', horizontalalignment='left', transform=ax.transAxes)

        '''
        dummy_str = 'TRYING TO COVER EXISTING COLORBAR'
        ax.annotate(dummy_str, xy=(0.5, -0.1), xycoords=ax.transAxes, fontsize=28,
                    va="center", ha="center", color='white',
                    bbox=dict(boxstyle="square",fc='white',ec='black'), zorder=50)

        fig.texts.append(ax.texts.pop())
        '''

    if prelim:
        ax.text(0.99, 0.95, prelim_str, horizontalalignment='right', color='red', fontsize=12, weight='bold', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.99,boxstyle='square,pad=0.2'))
        fig.texts.append(ax.texts.pop())
    

    ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
    ax.text(0.55, 0.05, label_str,  horizontalalignment='left', weight='bold', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.99,boxstyle='square,pad=0.2'))

    fig.texts.append(ax.texts.pop())

    valid_str  = valid_beg.strftime('Valid: %HZ %m/%d/%y')+valid_end.strftime(' to %HZ %m/%d/%y')
    plt.text(1, 1.01, valid_str, fontweight='bold', horizontalalignment='right', transform=ax.transAxes)


    # Set output filename
    if domain == 'dailyzoom':
        outfile = 'spc_'+str.upper(field)+'_dailyzoom_'+valid_beg.strftime('%Y%m%d%H')+'-'+valid_end.strftime('%Y%m%d%H')
    else:
        outfile = 'spc_'+str.upper(field)+'_'+valid_beg.strftime('%Y%m%d%H')+'-'+valid_end.strftime('%Y%m%d%H')

    plt.savefig(NOSCRUB_DIR+'/'+outfile+'.png',bbox_inches='tight')
    plt.close()



def main():

   pool = multiprocessing.Pool(len(plots))
   pool.map(plot_fields,plots)



main()
