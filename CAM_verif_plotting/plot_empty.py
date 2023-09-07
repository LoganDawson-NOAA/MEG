#!/usr/bin/env python
import netCDF4
import numpy as np
import os, sys, datetime, time, subprocess
import re, csv, glob
import multiprocessing, itertools, collections
import matplotlib
import matplotlib.image as image
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits
mpl_toolkits.__path__.append('/gpfs/dell2/emc/modeling/noscrub/gwv/py/lib/python/basemap-1.2.1-py3.6-linux-x86_64.egg/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap, cm
import dawsonpy

spc_rep_date = str(sys.argv[1])

# Get machine
machine, hostname = dawsonpy.get_machine()

# Set up working directory
if machine == 'WCOSS':
    pass
elif machine == 'WCOSS_C':
    pass
elif machine == 'WCOSS_DELL_P3':
    NOSCRUB_DIR = '/gpfs/dell2/emc/verification/noscrub/'+os.environ['USER']+'/CAM_verif/surrogate_severe/point2grid'
#   METOUT_DIR  = '/gpfs/dell2/ptmp/Logan.Dawson/CAM_verif/METplus/metplus.out/surrogate_severe/point2grid'


YYYY = int('20'+spc_rep_date[0:2])
MM   = int(spc_rep_date[2:4])
DD   = int(spc_rep_date[4:6])

valid_beg = datetime.datetime(YYYY,MM,DD,12,00,00)
valid_end = valid_beg + datetime.timedelta(hours=24)



ppf_fil = NOSCRUB_DIR+'/'+valid_beg.strftime('%Y%m')+'/PracticallyPerfect_'+valid_beg.strftime('%Y%m%d%H')+'-'+valid_end.strftime('%Y%m%d%H')+'_G211.nc'
nc = netCDF4.Dataset(ppf_fil,'r')
grid_lats = nc.variables['lat'][:]
grid_lons = nc.variables['lon'][:]
ppf_data = nc.variables['LSR_PPF'][:]*0.
nc.close()

lsr_fil = NOSCRUB_DIR+'/'+valid_beg.strftime('%Y%m')+'/LocalStormReports_'+valid_beg.strftime('%Y%m%d%H')+'-'+valid_end.strftime('%Y%m%d%H')+'_G211.nc'
nc = netCDF4.Dataset(lsr_fil,'r')
lsr_data = nc.variables['Fscale_mask'][:]*0.
nc.close()

print(np.min(lsr_data),np.max(lsr_data))
print(np.min(ppf_data),np.max(ppf_data))

large_domains = ['conus']
domains = ['conus']
fields = ['ppf']

plots = [n for n in itertools.product(domains,fields)]
print(plots)




def plot_fields(plot):

    thing = np.asarray(plot)
    domain = thing[0]
    field = thing[1]


    fig = plt.figure(figsize=(10.9,8.9))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    im = image.imread('/gpfs/dell2/emc/verification/save/Logan.Dawson/noaa.png')


    if domain == 'conus':
        llcrnrlon=-121.0
        llcrnrlat=21.0
        urcrnrlon=-62.6
        urcrnrlat=49.0
        xscale=0.15
        yscale=0.2

        m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,\
                    resolution='i',projection='lcc',\
                    lat_1=32.,lat_2=46.,lon_0=-101.,area_thresh=1000.,ax=ax)

    m.drawcoastlines()
    m.drawstates(linewidth=0.75)
    m.drawcountries()

    if domain in large_domains:
        latlongrid = 10.
        parallels = np.arange(-90.,91.,latlongrid)
        m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
        meridians = np.arange(0.,360.,latlongrid)
        m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    else:
        m.drawcounties(linewidth=0.2, color='k')


    dx = 81.271
    x, y = m(grid_lons,grid_lats)
    x_shift = x - (dx/2.)
    y_shift = y - (dx/2.)

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

      # fill = m.pcolormesh(x_shift,y_shift,ppf_data,cmap=cm,vmin=2,norm=norm,ax=ax)
      # fill.cmap.set_under('white',alpha=0.)
      # fill.cmap.set_over('orchid')
      # cbar = m.colorbar(fill,ax=ax,ticks=clevs,location='bottom',pad=0.05,shrink=0.75,aspect=20,extend='max')

        fill = m.contourf(grid_lons,grid_lats,ppf_data,clevs,latlon=True,cmap=cm,norm=norm,extend='max')
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
#       cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
#       cbar.ax.tick_params(labelsize=10)
#       cbar.set_label('%',color='white')

        label_str = 'No Data Available for Current Selections'

        plt.text(0, 1.06, 'SPC Local Storm Reports', fontweight='bold', color='white', horizontalalignment='left', transform=ax.transAxes)
        plt.text(0, 1.01, '(regridded to 81-km NCEP Grid 211)', fontweight='bold', color='white', horizontalalignment='left', transform=ax.transAxes)


    elif field == 'lsr':

        clevs = [1,1000]
        colorlist = ['black']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        fill = m.pcolormesh(x_shift,y_shift,lsr_data,cmap=cm,vmin=1,norm=norm,ax=ax)
        fill.cmap.set_under('white',alpha=0.)

        label_str = 'Number of OSRs: '+str(np.sum(lsr_data))+'\n(on 81-km grid)'

        plt.text(0, 1.06, 'SPC Local Storm Reports', fontweight='bold', horizontalalignment='left', transform=ax.transAxes)
        plt.text(0, 1.01, '(regridded to 81-km NCEP Grid 211)', fontweight='bold', horizontalalignment='left', transform=ax.transAxes)


    ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)
    ax.text(0.5, 0.5, label_str,  horizontalalignment='center', color='red', fontsize=15, weight='bold', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))



    dummy_levs = [500,1000]
    dummy_list = ['white']
    dummy_cm = matplotlib.colors.ListedColormap(dummy_list)
    dummy_norm = matplotlib.colors.BoundaryNorm(dummy_levs, dummy_cm.N)

    dummy = m.contourf(grid_lons,grid_lats,ppf_data,dummy_levs,latlon=True,cmap=dummy_cm,extend='max')
    dummy.cmap.set_over('white')

    cbar = plt.colorbar(dummy,ax=ax,ticks=dummy_levs,orientation='horizontal',pad=0.04,shrink=0.67,aspect=20,extend='max',drawedges=False)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('%',color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.xaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')

    '''
    dummy_str = 'TRYING TO COVER EXISTING COLORBAR'
    ax.annotate(dummy_str, xy=(0.5, -0.1), xycoords=ax.transAxes, fontsize=28,
                  va="center", ha="center", color='white',
                  bbox=dict(boxstyle="square",fc='white',ec='white'), zorder=50)

    fig.texts.append(ax.texts.pop())
    '''

    valid_str  = 'SPC Local Storm Reports'
    plt.text(1, 1.01, valid_str, fontweight='bold', color='white', horizontalalignment='right', transform=ax.transAxes)
    plt.text(1, 1.06, valid_str, fontweight='bold', color='white', horizontalalignment='right', transform=ax.transAxes)

    plt.savefig('nodata.png',bbox_inches='tight')
    plt.close()



def main():

   pool = multiprocessing.Pool(len(plots))
   pool.map(plot_fields,plots)



main()
