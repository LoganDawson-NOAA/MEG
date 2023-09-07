#!/usr/bin/env python
import os, sys, datetime, time, subprocess
import re, csv, glob
import numpy as np
import netCDF4
import multiprocessing, itertools, collections
import matplotlib
import matplotlib.image as image
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits
mpl_toolkits.__path__.append('/gpfs/dell2/emc/modeling/noscrub/gwv/py/lib/python/basemap-1.2.1-py3.6-linux-x86_64.egg/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import colors as c
import dawsonpy

# Read in date info
VALID_DATETIME = str(sys.argv[1])

YYYY = int(VALID_DATETIME[0:4])
MM   = int(VALID_DATETIME[4:6])
DD   = int(VALID_DATETIME[6:8])

valid_end = datetime.datetime(YYYY,MM,DD,12,00,00)


# Read in model to plot
models= [str(sys.argv[2])]


# Get machine
machine, hostname = dawsonpy.get_machine()

# Set up working directory
if machine == 'WCOSS':
    pass
elif machine == 'WCOSS_C':
    pass
elif machine == 'WCOSS_DELL_P3':
    DATA_HEAD = '/gpfs/dell2/ptmp/'+os.environ['USER']+'/CAM_verif/METplus/metplus.out/surrogate_severe'
    NOSCRUB_HEAD = '/gpfs/dell2/emc/verification/noscrub/'+os.environ['USER']+'/CAM_verif/surrogate_severe'
    CASE_DIR    = '/gpfs/dell2/ptmp/'+os.environ['USER']+'/case/plots'


# Create output directories
OUTPUT_DIR = CASE_DIR+'/pcp_combine/'+VALID_DATETIME
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_DIR = CASE_DIR+'/sspf/'+VALID_DATETIME
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Model info
model_strings = { 
    'CONUSARW'   : 'HiResW ARW',
    'CONUSARW2'  : 'HiResW ARW2',
    'CONUSFV3'   : 'HiResW FV3',
    'CONUSHREF'  : 'HREFv3',
    'CONUSNEST'  : 'NAM Nest',
    'CONUSNMMB'  : 'HiResW NMMB',
    'FV3LAM'     : 'FV3 LAM',
    'FV3LAMDA'   : 'FV3 LAM-DA',
    'FV3LAMDAX'  : 'FV3 LAM-DA-X',
    'FV3LAMX'    : 'FV3 LAM-X',
    'HRRR'       : 'HRRR'}

# List of all models (for UH25 and UH03)
#models = ['CONUSARW','CONUSARW2','CONUSFV3','CONUSNEST','CONUSHREF','CONUSHREFX','CONUSNMMB','FV3LAM','FV3LAMDA','FV3LAMDAX','FV3LAMX','HRRR']

# List of models with VV data
relv_models = ['FV3LAM','FV3LAMDA','FV3LAMDAX','FV3LAMX','HRRR']

# List of models on HiResW output grid
hrw_grid = ['CONUSARW','CONUSARW2','CONUSFV3','CONUSHREF','CONUSHREFX','CONUSNMMB']


arw_models = ['CONUSARW','CONUSARW2','HRRR']
nmmb_models = ['CONUSNEST']
fv3_models = ['CONUSFV3','RRFS_A']

large_domains = ['conus']

global domains, accum_length
domains = ['conus']

# Set accumulation length and valid begin time
accum_length = '24'
valid_beg = valid_end + datetime.timedelta(hours=-int(accum_length))


ppf_fil = NOSCRUB_HEAD+'/point2grid/'+valid_beg.strftime('%Y%m')+'/PracticallyPerfect_'+\
          valid_beg.strftime('%Y%m%d%H')+'-'+valid_end.strftime('%Y%m%d%H')+'_G211.nc'
if os.path.exists(ppf_fil):
    nc = netCDF4.Dataset(ppf_fil,'r')
    grid_lats = nc.variables['lat'][:]
    grid_lons = nc.variables['lon'][:]
    ppf_data = nc.variables['LSR_PPF'][:]*100.
    nc.close()

    max_ind = np.ma.MaskedArray.argmax(ppf_data)
    ppf_max_ind = np.unravel_index(max_ind,ppf_data.shape)
    ppf_max_lat = grid_lats[ppf_max_ind]
    ppf_max_lon = grid_lons[ppf_max_ind]

    domains.extend(['dailyzoom'])

domains = ['CPL']


mask_file = '/gpfs/dell2/emc/verification/noscrub/'+os.environ['USER']+'/CAM_verif/masks/G211/CONUS_G211.nc'
nc = netCDF4.Dataset(mask_file,'r')
conus_mask = nc.variables['CONUS'][:]




def main(model, domain):

    if model in relv_models: 
        fields = ['MXUPHL25_A24','MXUPHL03_A24','MXUPHL02_A24',
                  'RELV02_A24','RELV01_A24',
                  'MXUPHL25_A24_prob_HWT','MXUPHL25_A24_prob_HSLC',
                  'MXUPHL03_A24_prob_HWT','MXUPHL03_A24_prob_HSLC',
                  'MXUPHL02_A24_prob_HWT','MXUPHL02_A24_prob_HSLC',
                  'RELV02_A24_prob_HWT',  'RELV02_A24_prob_HSLC',
                  'RELV01_A24_prob_HWT',  'RELV01_A24_prob_HSLC']
    else:
        fields = ['MXUPHL25_A24','MXUPHL03_A24','MXUPHL02_A24',
                  'MXUPHL25_A24_prob_HWT','MXUPHL25_A24_prob_HSLC',
                  'MXUPHL03_A24_prob_HWT','MXUPHL03_A24_prob_HSLC',
                  'MXUPHL02_A24_prob_HWT','MXUPHL02_A24_prob_HSLC']

    fields = ['MXUPHL25_A24','MXUPHL25_A24_prob_HWT']

#   plots = [n for n in itertools.product(domains,fields)]
#   print(plots)

#   pool = multiprocessing.Pool(len(plots))
#   pool.map(plot_fields,plots)

    pool = multiprocessing.Pool(len(fields))
    pool.map(plot_fields,fields)




#def plot_fields(plot):
def plot_fields(field):


#   thing = np.asarray(plot)
#   domain = thing[0]
#   field = thing[1]
#   domain = domains[0]

    # Define GRIB code used in filenames
    if 'MXUPHL' in field:
        grib_code = 'MXUPHL'
    elif 'RELV' in field:
        grib_code = 'RELV'


    if 'prob' in field:
        sub_dir = 'sspf'
        glob_str = grib_code+'_SSPF'

        dx = 81.271
    else:
        sub_dir = 'pcp_combine'
        glob_str = grib_code+'_A'+accum_length

        if model in hrw_grid:
            dx = 5.079
        else:
            dx = 3.0


    # Define output directory where plots will be saved
    OUTPUT_DIR = CASE_DIR+'/'+sub_dir+'/'+VALID_DATETIME

    # Change to directory where accumulation data exist
    DATA_DIR = NOSCRUB_HEAD+'/'+sub_dir+'/'+VALID_DATETIME
    os.chdir(DATA_DIR)


    # Build list of available files for a specific model
    filelist = [f for f in glob.glob(DATA_DIR+'/'+model+'.*'+glob_str+'*-'+VALID_DATETIME+'.*.nc')]

    # Loop over files
    for fil in sorted(filelist):

        # Read forecast hour from filename
        fhr = int(fil[-5:-3])
 
        # Use forecast hour to define init_time 
        init_time = valid_end + datetime.timedelta(hours=-int(fhr)) 

        # Define string for plot information
        init_str = init_time.strftime('Init: %HZ %d %b %Y')
        cyc = init_time.strftime('%Y%m%d.t%Hz.')
       #valid_str  = valid_beg.strftime('Valid: %HZ %m/%d/%y')+valid_end.strftime('Valid: %HZ %m/%d/%y (F'+str(fhr).zfill(2)+')')
        valid_str  = valid_end.strftime('Valid: %HZ %m/%d/%y (F'+str(fhr).zfill(2)+')')


        nc = netCDF4.Dataset(fil,'r')
        grid_lats = nc.variables['lat'][:]
        grid_lons = nc.variables['lon'][:]



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
        elif domain == 'CPL':
            llcrnrlon=-105.0
            llcrnrlat=31.5
            urcrnrlon=-87.5
            urcrnrlat=45.0
            lat_1=25.
            lat_2=46.
            lon_0=-97.5
            xscale=0.15
            yscale=0.2
        elif domain == 'dailyzoom':
            llcrnrlon=ppf_max_lon-12
            llcrnrlat=ppf_max_lat-6
            urcrnrlon=ppf_max_lon+12
            urcrnrlat=ppf_max_lat+6
            xscale=0.17
            yscale=0.2


        print('plotting '+model+' '+field+' on '+domain+' domain at F'+str(fhr).zfill(2))

        m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,\
                    resolution='i',projection='lcc',\
                    lat_1=lat_1,lat_2=lat_2,lon_0=lon_0,area_thresh=1000.,ax=ax)

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
            latlongrid = 5.
            if 'prob' in field:
                m.drawcounties(linewidth=0.2, color='k',zorder=19)


        x, y = m(grid_lons,grid_lats)
        x_shift = x - (dx/2.)
        y_shift = y - (dx/2.)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xmax = int(round(xmax))
        ymax = int(round(ymax))

        if sub_dir == 'pcp_combine':

            data = nc.variables[field][:]

            if field[0:6] == 'MXUPHL':

                clevs = [25,50,75,100,125,150,175,200,250,300,400]
                units = '$\mathregular{m^{2}}$ $\mathregular{s^{-2}}$'
                label_units = 'm^2/s^2'
                field_str = 'UH'

            elif field[0:4] == 'RELV':

                clevs = [0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015]
                units = '$\mathregular{s^{-1}}$'
                label_units = '1/s'
                field_str = 'Vertical Vorticity'


            colorlist = ['turquoise','dodgerblue','lime','limegreen','green', \
                         'yellow','gold','darkorange','red','firebrick']
            cm = matplotlib.colors.ListedColormap(colorlist)
            norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

            fill = m.pcolormesh(x_shift,y_shift,data,cmap=cm,vmin=clevs[0],norm=norm,ax=ax)
            fill.cmap.set_under('white',alpha=0.)
            fill.cmap.set_over('fuchsia')
            cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.67,aspect=20,extend='max')
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label(label_units)

            label_str = 'Max Value = '+str(round(np.max(data),3))+' '+units

            plt.text(0, 1.06, model_strings[model], fontweight='bold', horizontalalignment='left', transform=ax.transAxes)
            plt.text(0, 1.01, accum_length+'-h Max '+field[-6]+'-'+field[-5]+' km '+field_str, fontweight='bold', horizontalalignment='left', transform=ax.transAxes)



        elif sub_dir == 'sspf':

            if field[0:6] == 'MXUPHL':
                level_str = field[0:8]
                field_str = 'UH'
                units = '$\mathregular{m^{2}}$ $\mathregular{s^{-2}}$'

            elif field[0:4] == 'RELV':
                level_str = field[0:6]
                field_str = 'Vertical Vorticity'
                units = '$\mathregular{s^{-1}}$'

            if field[-3:] == 'HWT':
                if model[0:9] == 'CONUSHREF':
                    thresh_str = '2018 HWT Thresholds by Dycore'
                else:
                    if model in arw_models: 
                        thresh_str = 'Threshold >= 75 '+units
                    elif model in fv3_models: 
                        thresh_str = 'Threshold >= 160 '+units
                    elif model in nmmb_models: 
                        thresh_str = 'Threshold >= 100 '+units
            elif field[-4:] == 'HSLC':
                if model[0:9] == 'CONUSHREF':
                    thresh_str = 'Test High-Shear/Low-CAPE Threshold'
                else:
                    thresh_str = 'Test High-Shear/Low-CAPE Threshold >= '+os.environ[level_str+'_THRESH2']+' '+units
            

            data = nc.variables[field][:]*100.
            land_data = data[conus_mask == 1.]

            sspf_cov_step1 = np.sum(x > 2 for x in land_data)
            sspf_cov = round(np.sum(sspf_cov_step1)*81.271**2,0)
            sspf_max = round(np.max(land_data),2)

            clevs = [2,5,10,15,30,45,60,70,80,90,95]
            colorlist = ['blue','dodgerblue','cyan','limegreen','chartreuse','yellow', \
                         'orange','red','darkred','purple']

            cm = matplotlib.colors.ListedColormap(colorlist)
            norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

            fill = m.contourf(grid_lons,grid_lats,data,clevs,latlon=True,cmap=cm,norm=norm,extend='max')
            fill.cmap.set_under('white',alpha=0.)
            fill.cmap.set_over('orchid')
            cbar = plt.colorbar(fill,ax=ax,ticks=clevs,orientation='horizontal',pad=0.04,shrink=0.67,aspect=20)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('%')

            label_str = 'Surrogate Severe Max Value = '+str(sspf_max)+'%\n2% Coverage: '+str(sspf_cov)+' km^2'

            plt.text(0, 1.06, model_strings[model]+' SSPF based on '+\
                     accum_length+'-h Max '+level_str[-2]+'-'+level_str[-1]+' km '+field_str,
                     fontweight='bold', horizontalalignment='left', transform=ax.transAxes)

            plt.text(0, 1.01, 'using '+thresh_str,fontweight='bold', horizontalalignment='left', transform=ax.transAxes)

    
#       ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)
#       ax.text(0.60, 0.05, label_str,  horizontalalignment='left', weight='bold', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
#       fig.texts.append(ax.texts.pop())

        plt.text(1, 1.06, init_str, fontweight='bold', horizontalalignment='right', transform=ax.transAxes)
        plt.text(1, 1.01, valid_str, fontweight='bold', horizontalalignment='right', transform=ax.transAxes)

        # Set output filename
        if domain == 'dailyzoom':
            outfile = model+'.'+cyc+field+'_'+domain+'.'+fil[-28:-3] 
        else: 
            outfile = model+'.'+cyc+field+'.'+fil[-28:-3] 

        plt.savefig(OUTPUT_DIR+'/'+outfile+'.png',bbox_inches='tight')
        plt.close()


global model, domain
print(models)
for model in models:
    print(model)
    for domain in domains:
        main(model,domain)
