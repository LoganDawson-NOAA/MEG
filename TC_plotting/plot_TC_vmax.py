#!/usr/bin/env python

# Run as:
# python plot_TC_vmax.py $TC_name/ID/year
# python plot_TC_vmax.py FlorenceAL062018
#

import numpy as np
import os, sys, datetime, time, subprocess
import re, csv, glob
import multiprocessing, itertools, collections
import scipy, ncepy

import matplotlib
import matplotlib.image as image
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib import colors as c
matplotlib.use('Agg')

import cartopy



# Get TC name and number
try:
   TC = str(sys.argv[1])
except IndexError:
   TC = None

if TC is None:
   print('Enter TC name, number, and year as one string')
   print('Example: FlorenceAL062018')
   TC = input('Enter TC name/number/year: ')

TC_name = TC[:-8]
TC_number = TC[-8:-4]
YYYY = TC[-4:]
print(TC_name, TC_number, YYYY)


# Option to make special HAFS comparison graphics
try:
    dummy = str(sys.argv[2])
    do_hafs = True
    print('Plotting HAFS comparison images')
except IndexError:
    dummy = None
    do_hafs = False



# Set path and create graphx directory (if not already created)
MEG_DIR = os.getcwd()

BDECK_DIR = '/lfs/h1/ops/prod/dcom/nhc/atcf-noaa/btk'

if do_hafs:
    DATA_DIR = os.path.join('/lfs/h2/emc/vpppg/noscrub/',os.environ['USER'],'MEG', TC_name, 'data','hafs')
    GRAPHX_DIR = os.path.join('/lfs/h2/emc/ptmp',os.environ['USER'],'MEG', TC_name, 'graphx','hafs')
else:
    DATA_DIR = os.path.join('/lfs/h2/emc/vpppg/noscrub/',os.environ['USER'],'MEG', TC_name, 'data')
    GRAPHX_DIR = os.path.join('/lfs/h2/emc/ptmp',os.environ['USER'],'MEG', TC_name, 'graphx')

if os.path.exists(DATA_DIR):
   if not os.path.exists(GRAPHX_DIR):
      os.makedirs(GRAPHX_DIR)
else:
   raise NameError('data for '+TC_name+' not found')


#OUT_DIR = os.path.join(GRAPHX_DIR, cycle)
OUT_DIR = GRAPHX_DIR
if not os.path.exists(OUT_DIR):
      os.makedirs(OUT_DIR)


try:
   valid_time
except NameError:
   valid_time = None


# Get list of GFS cycles to get max number of cycles
filelist1 = [f for f in glob.glob(DATA_DIR+'/'+str.lower(TC_name)+'_gfs_'+str(YYYY)+'*csv')]
max_cycles_unordered = [filelist1[x][-14:-4] for x in range(len(filelist1))]
max_cycles_int = [int(x) for x in max_cycles_unordered]
max_cycles_int.sort()
max_cycles = [str(x) for x in max_cycles_int]
max_ncycles = len(max_cycles)


# Get list of cycles based on matching files in DATA_DIR
filelist = [f for f in glob.glob(DATA_DIR+'/'+str.lower(TC_name)+'_*_vmaxlist.csv')]



mlats=[]
mlons=[]
mpres=[]
mvmax=[]
#mrmw=[]
mcycles=[]
mnames=[]
mcolors=[]
models=['HWRF','HMON','GFS','FV3GFS','EC','UKMet']
models=['HWRF','HMON','GFS','GFSO','EC','UK']

if do_hafs:
    models=['HWRF','HMON','HF3A','HF3S']
    model_strings=['HWRF','HMON','HAFS-A','HAFS-S']
else:
    models=['HWRF','HMON','GFS','ECMO','UK']
    model_strings=['HWRF','HMON','GFS','EC','UKM']


#max_ncycles = 0
k = 0
for model_str in models:

   vmax_file = DATA_DIR+'/'+str.lower(TC_name)+'_'+str.lower(model_str)+'_vmaxlist.csv'

   color_list=[]

   if os.path.exists(vmax_file):
      with open(vmax_file,'r') as f:
         reader=csv.reader(f)
         i = 0
         for row in reader:
            if i == 0:
               cycle_list = [datetime.datetime.strptime(x,"%Y%m%d%H") for x in row]

               for cycle in cycle_list:
                  if cycle.strftime("%d") == '05' or cycle.strftime("%d") == '06':
                     color_list.append('blue')
                  elif cycle.strftime("%d") == '07':
                     color_list.append('green')
                  elif cycle.strftime("%d") == '08':
                     color_list.append('yellow')
                  elif cycle.strftime("%d") == '09':
                     color_list.append('orange')
                  elif cycle.strftime("%d") == '10':
                     color_list.append('red')

            elif i == 1:
               vmax_list = [float(x) for x in row]
            i += 1
 
#  if len(cycle_list) > max_ncycles:
#      max_ncycles = len(cycle_list)
#      max_ncycles_ind = k 

   mvmax.append(vmax_list)
   mcycles.append(cycle_list)
   mcolors.append(color_list)

   k += 1


#print(mvmax[0])
#print(mcycles[0])


cmap=matplotlib.cm.get_cmap('YlGnBu')

values = []
print(max_ncycles)
for i in range(max_ncycles):
   values.append(cmap(float(i+1)/float(max_ncycles+1)))
color_dict2 = dict(zip(max_cycles,values))
#color_dict2 = dict(zip(mcycles[max_ncycles_ind],values))

color_dict = {
   "2018100518": "powderblue",
   "2018100600": "skyblue",
   "2018100606": "dodgerblue",
   "2018100612": "blue",
   "2018100618": "navy",

   "2018100700": "lawngreen",
   "2018100706": "limegreen",
   "2018100712": "forestgreen",
   "2018100718": "darkgreen",

   "2018100800": "khaki",
   "2018100806": "yellow",
   "2018100812": "gold",
   "2018100818": "orange",

   "2018100900": "lightsalmon",
   "2018100906": "red",
   "2018100912": "firebrick",
   "2018100918": "maroon",

   "2018101000": "pink",
   "2018101006": "hotpink",
   "2018101012": "magenta",
   "2018101018": "darkmagenta",


   "2019071000": "skyblue",
   "2019071006": "dodgerblue",
   "2019071012": "blue",
   "2019071018": "navy",

   "2019071100": "lawngreen",
   "2019071106": "limegreen",
   "2019071112": "forestgreen",
   "2019071118": "darkgreen",

   "2019071200": "khaki",
   "2019071206": "yellow",
   "2019071212": "gold",
   "2019071218": "orange",

   "2019071300": "lightsalmon",
   "2019071306": "red",
   "2019071312": "firebrick",
   "2019071318": "maroon",
}


ovmax=0.
with open(BDECK_DIR+'/b'+str.lower(TC_number)+str(YYYY)+'.dat','r') as f:
   reader = csv.reader(f)
   for row in reader:
    # if row[10].replace(" ","")!='FAKE STRING':
    # if row[10].replace(" ","")=='TD' or row[10].replace(" ","")=='TS' or row[10].replace(" ","")=='HU':
    # if row[10].replace(" ","")=='TD' or row[10].replace(" ","")=='TS' or row[10].replace(" ","")=='HU' or row[10].replace(" ","")=='EX':
      if row[10].replace(" ","")=='DB' or row[10].replace(" ","")=='LO' or row[10].replace(" ","")=='TD' or row[10].replace(" ","")=='TS' or row[10].replace(" ","")=='HU' or row[10].replace(" ","")=='EX':
         if row[11].replace(" ","")=='34' or row[11].replace(" ","")=='0':
            if float(row[8]) > ovmax:
               ovmax = float(row[8])
            



def plot_vmax():

   print('plotting vmax scatter')

   fig = plt.figure(figsize=(9,8))

   for i in range(len(models)):
    # label_str = str.upper(models[i])
      if str.upper(models[i]) == 'GFS':
         models[i] = 'GFS'
      elif str.upper(models[i]) == 'GFSO':
         models[i] = 'GFSv14'

      x = [i+1 for x in range(len(mvmax[i]))]
      y = mvmax[i] 

      colors_list = [color_dict2[cycle.strftime("%Y%m%d%H")] for cycle in mcycles[i]]
    # colors_list = [color_dict2[cycle] for cycle in mcycles[i]]
      
    # plt.scatter(x, y, s=50 , c='k', alpha=0.5)
    # plt.scatter(x, y, s=50 , c=mcolors[i])
      plt.scatter(x, y, s=150 , c=colors_list, edgecolors='k',zorder=20)

#  plt.plot(plot_ovmax, '-', color='black', label='BEST', linewidth=2.)


   xlen = len(models) 
#  x = np.arange(0,xlen+1,1)

   if do_hafs:
       plt.axis([0,xlen+1,55,165])
   else:
       plt.axis([0,xlen+1,5,160])
   plt.axhline(y=ovmax,xmin=0,xmax=xlen+1.5,color='magenta',linewidth=2,linestyle='--')   # observed vmax line
   plt.axhline(y=34,xmin=0,xmax=xlen+1.5,color='k',linewidth=2)
   plt.axhline(y=64,xmin=0,xmax=xlen+1.5,color='k',linewidth=2)
   plt.axhline(y=83,xmin=0,xmax=xlen+1.5,color='k',linewidth=2)
   plt.axhline(y=96,xmin=0,xmax=xlen+1.5,color='k',linewidth=2)
   plt.axhline(y=113,xmin=0,xmax=xlen+1.5,color='k',linewidth=2)
   plt.axhline(y=137,xmin=0,xmax=xlen+1.5,color='k',linewidth=2)
   plt.axhspan(137, 200, facecolor='0.1', alpha=0.5)
   plt.axhspan(113, 137, facecolor='0.15', alpha=0.5)
   plt.axhspan(96, 113, facecolor='0.2', alpha=0.5)
   plt.axhspan(83, 96, facecolor='0.25', alpha=0.5)
   plt.axhspan(64, 83, facecolor='0.3', alpha=0.5)
   plt.axhspan(34, 64, facecolor='0.4', alpha=0.5)
   plt.axhspan(0, 34, facecolor='0.5', alpha=0.5)

   plt.xticks(np.arange(1,xlen+1,1),model_strings,weight='bold')
   plt.ylabel('Maximum Sustained 10-m Wind (kts)',fontweight='bold')

   plt.grid(True)

#  plt.legend(loc="upper right", ncol=5)

#  titlestr = 'Hurricane '+TC_name+' Maximum 10-m Wind Forecast by Initialization \n'+ \
#             cycle_date.strftime('%HZ %d %b Initializations')+' valid through '+final_valid_date.strftime('%HZ %d %b %Y')
   titlestr = 'Hurricane '+TC_name+' ('+YYYY+') - Maximum Intensity Forecast by Initialization'
   plt.title(titlestr, fontweight='bold')

   fname = str.lower(TC_name)+'_vmax'

   plt.savefig(OUT_DIR+'/'+fname+'.png',bbox_inches='tight')
   plt.close()




plot_vmax()


