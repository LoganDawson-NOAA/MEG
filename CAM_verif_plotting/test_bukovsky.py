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


if grid != 'G211':
    east_fil = NOSCRUB_DIR+'/Bukovsky_'+grid+'_CONUS_East.nc'
    nc = netCDF4.Dataset(east_fil,'r')
    conus_east = nc.variables['CONUS_East'][:]
    nc.close()

    west_fil = NOSCRUB_DIR+'/Bukovsky_'+grid+'_CONUS_West.nc'
    nc = netCDF4.Dataset(west_fil,'r')
    conus_west = nc.variables['CONUS_West'][:]
    nc.close()

    central_fil = NOSCRUB_DIR+'/Bukovsky_'+grid+'_CONUS_Central.nc'
    nc = netCDF4.Dataset(central_fil,'r')
    conus_central = nc.variables['CONUS_Central'][:]
    nc.close()

    south_fil = NOSCRUB_DIR+'/Bukovsky_'+grid+'_CONUS_Sout.nc'
    nc = netCDF4.Dataset(south_fil,'r')
    conus_south = nc.variables['CONUS_South'][:]
    nc.close()




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



'''
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



if np.max(lsr_data) == 1:
    max_ind = np.ma.MaskedArray.argmax(ppf_data)
    ppf_max_ind = np.unravel_index(max_ind,ppf_data.shape)
    ppf_max_lat = grid_lats[ppf_max_ind]
    ppf_max_lon = grid_lons[ppf_max_ind]

    domains.extend(['dailyzoom'])
'''

dx = 0.5

n_conus = np.sum(conus)

fields = ['conus','opt4','opt1','opt2','opt3']
large_domains = ['conus']
domains = ['conus']

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
    elif domain == 'dailyzoom':
        llcrnrlon=ppf_max_lon-12
        llcrnrlat=ppf_max_lat-6
        urcrnrlon=ppf_max_lon+12
        urcrnrlat=ppf_max_lat+6
        print(llcrnrlat,llcrnrlon)
        print(urcrnrlat,urcrnrlon)
        xscale=0.17
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
        m.drawcounties(linewidth=0.2, color='k',zorder=19)


#   x, y = m(grid_lons,grid_lats)
    x, y = np.meshgrid(grid_lons,grid_lats)
    lon_shift = grid_lons - (dx/2.)
    lat_shift = grid_lats - (dx/2.)
    x_shift, y_shift = np.meshgrid(lon_shift,lat_shift)


    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))



    if field == 'conus':

        clevs = [1,1000]
        colorlist = ['red']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        fill = m.pcolormesh(x_shift,y_shift,conus,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')

        label_str = 'Number of CONUS grid cells: '+str(int(n_conus))


    elif field == 'opt1':


        clevs = [1,1000]
        colorlist = ['green']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        east = greatlakes + appalachia + midatlantic + northeast + southeast
        n_east = np.sum(east)
        east_cov = n_east/n_conus
        fill = m.pcolormesh(x_shift,y_shift,east,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
      # fill = m.contourf(x,y,east,clevs,latlon=True,cmap=cm,norm=norm,extend='max')


        clevs = [1,1000]
        colorlist = ['red']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        central = nplains + cplains + splains + prairie + deepsouth + mezquital
        n_central = np.sum(central)
        central_cov = n_central/n_conus
        fill = m.pcolormesh(x_shift,y_shift,central,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
      # fill = m.contourf(x,y,central,clevs,latlon=True,cmap=cm,norm=norm,extend='max')


        clevs = [1,1000]
        colorlist = ['purple']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        west = pacificnw + pacificsw + greatbasin + southwest + nrockies + srockies
        n_west = np.sum(west)
        west_cov = n_west/n_conus
        fill = m.pcolormesh(x_shift,y_shift,west,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
      # fill = m.contourf(x,y,west,clevs,latlon=True,cmap=cm,norm=norm,extend='max')


        label_str = 'Number of CONUS grid cells: '+str(int(n_conus))+'\n'+ \
                    'Number of West cells: '+str(int(n_west))+' ('+str(round(west_cov*100.,2))+'%)\n'+ \
                    'Number of Central cells: '+str(int(n_central))+' ('+str(round(central_cov*100.,2))+'%)\n'+ \
                    'Number of East cells: '+str(int(n_east))+' ('+str(round(east_cov*100.,2))+'%)'
                     

    elif field == 'opt2':


        clevs = [1,1000]
        colorlist = ['green']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        east = greatlakes + appalachia + midatlantic + northeast 
        n_east = np.sum(east)
        east_cov = n_east/n_conus
        fill = m.pcolormesh(x_shift,y_shift,east,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
      # fill = m.contourf(x,y,east,clevs,latlon=True,cmap=cm,norm=norm,extend='max')


        clevs = [1,1000]
        colorlist = ['red']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        central = nplains + cplains + splains + prairie + deepsouth + southeast
        n_central = np.sum(central)
        central_cov = n_central/n_conus
        fill = m.pcolormesh(x_shift,y_shift,central,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
      # fill = m.contourf(x,y,central,clevs,latlon=True,cmap=cm,norm=norm,extend='max')


        clevs = [1,1000]
        colorlist = ['purple']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        west = pacificnw + pacificsw + greatbasin + southwest + nrockies + srockies + mezquital
        n_west = np.sum(west)
        west_cov = n_west/n_conus
        fill = m.pcolormesh(x_shift,y_shift,west,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
      # fill = m.contourf(x,y,west,clevs,latlon=True,cmap=cm,norm=norm,extend='max')


        label_str = 'Number of CONUS grid cells: '+str(int(n_conus))+'\n'+ \
                    'Number of West cells: '+str(int(n_west))+' ('+str(round(west_cov*100.,2))+'%)\n'+ \
                    'Number of Central cells: '+str(int(n_central))+' ('+str(round(central_cov*100.,2))+'%)\n'+ \
                    'Number of East cells: '+str(int(n_east))+' ('+str(round(east_cov*100.,2))+'%)'
                     



    elif field == 'opt3':

        clevs = [1,1000]
        colorlist = ['blue']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        south = deepsouth + southeast
        n_south = np.sum(south)
        south_cov = n_south/n_conus
        fill = m.pcolormesh(x_shift,y_shift,south,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
  #     fill = m.contourf(x,y,south,clevs,latlon=True,cmap=cm,norm=norm,extend='max')


        clevs = [1,1000]
        colorlist = ['green']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        east = greatlakes + appalachia + midatlantic + northeast 
        n_east = np.sum(east)
        east_cov = n_east/n_conus
        fill = m.pcolormesh(x_shift,y_shift,east,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
      # fill = m.contourf(x,y,east,clevs,latlon=True,cmap=cm,norm=norm,extend='max')


        clevs = [1,1000]
        colorlist = ['red']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        central = nplains + cplains + splains + prairie + mezquital
        n_central = np.sum(central)
        central_cov = n_central/n_conus
        fill = m.pcolormesh(x_shift,y_shift,central,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
      # fill = m.contourf(x,y,central,clevs,latlon=True,cmap=cm,norm=norm,extend='max')


        clevs = [1,1000]
        colorlist = ['purple']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        west = pacificnw + pacificsw + greatbasin + southwest + nrockies + srockies
        n_west = np.sum(west)
        west_cov = n_west/n_conus
        fill = m.pcolormesh(x_shift,y_shift,west,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
      # fill = m.contourf(x,y,west,clevs,latlon=True,cmap=cm,norm=norm,extend='max')


        label_str = 'Number of CONUS grid cells: '+str(int(n_conus))+'\n'+ \
                    'Number of West cells: '+str(int(n_west))+' ('+str(round(west_cov*100.,2))+'%)\n'+ \
                    'Number of Central cells: '+str(int(n_central))+' ('+str(round(central_cov*100.,2))+'%)\n'+ \
                    'Number of East cells: '+str(int(n_east))+' ('+str(round(east_cov*100.,2))+'%)\n'+ \
                    'Number of South cells: '+str(int(n_south))+' ('+str(round(south_cov*100.,2))+'%)'
                     



    elif field == 'opt4':

        clevs = [1,1000]
        colorlist = ['blue']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        south = deepsouth + southeast
        n_south = np.sum(south)
        south_cov = n_south/n_conus
        fill = m.pcolormesh(x_shift,y_shift,south,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
  #     fill = m.contourf(x,y,south,clevs,latlon=True,cmap=cm,norm=norm,extend='max')


        clevs = [1,1000]
        colorlist = ['green']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        east = greatlakes + appalachia + midatlantic + northeast 
        n_east = np.sum(east)
        east_cov = n_east/n_conus
        fill = m.pcolormesh(x_shift,y_shift,east,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
      # fill = m.contourf(x,y,east,clevs,latlon=True,cmap=cm,norm=norm,extend='max')


        clevs = [1,1000]
        colorlist = ['red']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        central = nplains + cplains + splains + prairie
        n_central = np.sum(central)
        central_cov = n_central/n_conus
        fill = m.pcolormesh(x_shift,y_shift,central,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
      # fill = m.contourf(x,y,central,clevs,latlon=True,cmap=cm,norm=norm,extend='max')


        clevs = [1,1000]
        colorlist = ['purple']
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

        west = pacificnw + pacificsw + greatbasin + southwest + nrockies + srockies + mezquital
        n_west = np.sum(west)
        west_cov = n_west/n_conus
        fill = m.pcolormesh(x_shift,y_shift,west,cmap=cm,vmin=1,norm=norm,ax=ax,latlon=True)
        fill.cmap.set_under('white',alpha=0.)
        fill.cmap.set_over('orchid')
      # fill = m.contourf(x,y,west,clevs,latlon=True,cmap=cm,norm=norm,extend='max')


        label_str = 'Number of CONUS grid cells: '+str(int(n_conus))+'\n'+ \
                    'Number of West cells: '+str(int(n_west))+' ('+str(round(west_cov*100.,2))+'%)\n'+ \
                    'Number of Central cells: '+str(int(n_central))+' ('+str(round(central_cov*100.,2))+'%)\n'+ \
                    'Number of East cells: '+str(int(n_east))+' ('+str(round(east_cov*100.,2))+'%)\n'+ \
                    'Number of South cells: '+str(int(n_south))+' ('+str(round(south_cov*100.,2))+'%)'
                     


  #     fill = m.pcolormesh(x_shift,y_shift,south,cmap=cm,vmin=0.5,norm=norm,ax=ax)
  #     fill = m.pcolormesh(grid_lons,grid_lats,south,cmap=cm,vmin=1,norm=norm,ax=ax)
     #  fill.cmap.set_under('white',alpha=0.)
     #  fill.cmap.set_over('orchid')
  #     cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.75,aspect=20)
  #     cbar.ax.tick_params(labelsize=10)
  #     cbar.set_label('%')
  #     cbar.remove()




#       plt.text(0, 1.06, 'SPC Local Storm Reports', fontweight='bold', horizontalalignment='left', transform=ax.transAxes)
#       plt.text(0, 1.01, '(regridded to 81-km NCEP Grid 211)', fontweight='bold', horizontalalignment='left', transform=ax.transAxes)

    

  # ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)
    ax.text(0.05, 0.04, label_str,  horizontalalignment='left', weight='bold', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.99,boxstyle='square,pad=0.2'))

    outfile = field 

    plt.savefig(PTMP_DIR+'/'+outfile+'.png',bbox_inches='tight')
    plt.close()



def main():

   pool = multiprocessing.Pool(len(plots))
   pool.map(plot_fields,plots)



main()
