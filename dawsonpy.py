import re, os
import time
import matplotlib
import numpy as np
from scipy.ndimage.filters import minimum_filter, maximum_filter
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import cStringIO
#from PIL import Image


# Function to determine which machine we're on
# Code from Mallory Row (NCEP/EMC)
def get_machine():
    hostname = os.environ['HOSTNAME']
    if "machine" in os.environ:
        machine = os.environ['machine']
    else:
        if "MACHINE" in os.environ:
            machine = os.environ['MACHINE']
        else:
            theia_match  = re.match(re.compile(r"^tfe[0-9]{2}$"), hostname)
            hera_match   = re.match(re.compile(r"^hfe[0-9]{2}$"), hostname)
            tide_match   = re.match(re.compile(r"^t[0-9]{2}a[0-9]{1}$"), hostname)
            gyre_match   = re.match(re.compile(r"^g[0-9]{2}a[0-9]{1}$"), hostname)
            surge_match  = re.match(re.compile(r"^slogin[0-9]{1}$"), hostname)
            luna_match   = re.match(re.compile(r"^llogin[0-9]{1}$"), hostname)
            mars_match   = re.match(re.compile(r"^m[0-9]{2}[a-z]{1}[0-9]{1,2}$"), hostname)
            venus_match  = re.match(re.compile(r"^v[0-9]{2}[a-z]{1}[0-9]{1,2}$"), hostname)
            mars_match2  = re.match(re.compile(r"^m[0-9]{2}[a-z]{1}[0-9]{1,2}f$"), hostname)
            venus_match2 = re.match(re.compile(r"^v[0-9]{2}[a-z]{1}[0-9]{1,2}f$"), hostname)
            mars_match3  = re.match(re.compile(r"^m[0-9]{2}[a-z]{1}[0-9]{1,2}.ncep.noaa.gov$"), hostname)
            venus_match3 = re.match(re.compile(r"^v[0-9]{2}[a-z]{1}[0-9]{1,2}.ncep.noaa.gov$"), hostname)
            if hera_match:
                machine = 'HERA'
            elif tide_match or gyre_match:
                machine = 'WCOSS'
                if tide_match:
                    system = 'tide'
                elif gyre_match:
                    system = 'gyre'
            elif luna_match or surge_match:
                machine = 'WCOSS_C'
            elif mars_match or venus_match or mars_match2 or venus_match2 or mars_match3 or venus_match3:
                machine = 'WCOSS_DELL_P3'
                if mars_match or mars_match2 or mars_match3:
                    system = 'mars'
                elif venus_match or venus_match2 or venus_match3:
                    system = 'venus'
            else: 
                print("Cannot find match for "+hostname)
                exit(1)

    return machine, hostname



# Function to submit batch scripts
def submit_job(job_script):
    print('Submitting '+job_script)
    os.system('qsub '+job_script)
    time.sleep(5)



# Function to do multiple replacements using re.sub
# From: https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex-in-python
def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)

    if __name__ == "__main__":

        text = "Larry Wall is the creator of Perl"

        dict = {
            "Larry Wall" : "Guido van Rossum",
            "creator" : "Benevolent Dictator for Life",
            "Perl" : "Python",
        }

        print(multiple_replace(dict, text))


# Function to clear off old plottables but leave the map info
def clear_plotables(ax,keep_ax_lst,fig):
  if len(keep_ax_lst) == 0 :
    print("clear_plotables WARNING keep_ax_lst has length 0. Clearing ALL plottables including map info!")
  cur_ax_children = ax.get_children()[:]
  if len(cur_ax_children) > 0:
    for a in cur_ax_children:
      if a not in keep_ax_lst:
       # if the artist isn't part of the initial set up, remove it
        a.remove()

'''
# Function to compress and save image file
def compress_and_save(filename):
# plt.savefig(filename, format='png', bbox_inches='tight', dpi=150)
  ram = cStringIO.StringIO()
  plt.savefig(ram, format='png', bbox_inches='tight', dpi=150)
  ram.seek(0)
  im = Image.open(ram)
  im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
  im2.save(filename, format='PNG')
'''

# Function to determine which domains to plot
# For RAPv5/HRRRv4 retros (julaug2018, febmar2019, and may2019)
def get_domains(case, model_str):
    splains_cases   = ['apr30','aug13','mar09','may01','may02','may05','may07','may17','may18','may20','may23','may29']
    cplains_cases   = ['aug14','feb23','jul16','may04','may05','may06','may17','may21','may22','may23','may26','may27','may28','jul19','jul27','jul29']
    nplains_cases   = ['feb04','feb07','feb16','mar09','may15','jul16','jul18','jul27']
    midwest_cases   = ['aug05','feb05','feb07','feb11','feb12','feb23','may16','may19','may22','may26','may27','may28','may29','jul20']
    northeast_cases = ['aug03','aug13','feb12','mar03','may19','may23','may26','may28','may29','jul17','jul21','jul22','jul23','jul24','jul25','jul27']
    southeast_cases = ['aug01','aug02','aug03','feb22','feb23','mar03','mar09','may04','may05','may13','jul20','jul21','jul22','jul23','jul24','jul25']
    southwest_cases = ['feb02','feb03','feb04','feb05','feb14','feb21','jul16','jul17']
    northwest_cases = ['feb02','feb03','feb04','feb08','feb11','feb12','feb16','may16']

    if str.upper(model_str) != 'HRRR-AK' and str.upper(model_str) != 'RAP-AK':
        domains = ['conus']
        if str.lower(case) in splains_cases:
            domains.extend(['splains'])
        if str.lower(case) in cplains_cases:
            domains.extend(['cplains'])
        if str.lower(case) in nplains_cases:
            domains.extend(['nplains'])
        if str.lower(case) in midwest_cases:
            domains.extend(['midwest'])
        if str.lower(case) in northeast_cases:
            domains.extend(['northeast'])
        if str.lower(case) in southeast_cases:
            domains.extend(['southeast'])
        if str.lower(case) in northwest_cases:
            domains.extend(['northwest'])
        if str.lower(case) in southwest_cases:
            domains.extend(['southwest'])
    else:
        domains = ['alaska']
 
    return domains



def extrema(mat,mode='wrap',window=10):
  # From: http://matplotlib.org/basemap/users/examples.html

  """find the indices of local extrema (min and max)
  in the input array."""
  mn = minimum_filter(mat, size=window, mode=mode)
  mx = maximum_filter(mat, size=window, mode=mode)
  # (mat == mx) true if pixel is equal to the local max
  # (mat == mn) true if pixel is equal to the local in
  # Return the indices of the maxima, minima
  return np.nonzero(mat == mn), np.nonzero(mat == mx)


def plt_highs_and_lows(m,mat,lons,lats,mode='wrap',window=10,font=14):
  # From: http://matplotlib.org/basemap/users/examples.html
  # m is the map handle
  if isinstance(window,int) == False:
    raise TypeError("The window argument to plt_highs_and_lows must be an integer.") 
  x, y = m(lons, lats)
  local_min, local_max = extrema(mat,mode,window)
  xlows = x[local_min]; xhighs = x[local_max]
  ylows = y[local_min]; yhighs = y[local_max]
  lowvals = mat[local_min]; highvals = mat[local_max]
  # plot lows as red L's, with min pressure value underneath.
  xyplotted = []
  # don't plot if there is already a L or H within dmin meters.
  yoffset = 0.022*(m.ymax-m.ymin)
  dmin = yoffset
  for x,y,p in zip(xlows, ylows, lowvals):
    if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin:
        dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
        if not dist or min(dist) > dmin:
         #  plt.text(x,y,'L',fontsize=14,fontweight='bold',
            plt.text(x,y,'L',fontsize=font,fontweight='bold',
                    ha='center',va='center',color='r',zorder=10,clip_on=True)
         #  plt.text(x,y-yoffset,repr(int(p)),fontsize=9,zorder=10,
            plt.text(x,y-yoffset,repr(int(p)),fontsize=font,fontweight='bold',
                    ha='center',va='top',color='r',zorder=10,
                    bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)),clip_on=True)
            xyplotted.append((x,y))
  # plot highs as blue H's, with max pressure value underneath.
  xyplotted = []
  for x,y,p in zip(xhighs, yhighs, highvals):
    if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin:
        dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
        if not dist or min(dist) > dmin:
            plt.text(x,y,'H',fontsize=font,fontweight='bold',
                    ha='center',va='center',color='b',zorder=10,clip_on=True)
            plt.text(x,y-yoffset,repr(int(p)),fontsize=font,fontweight='bold',
                    ha='center',va='top',color='b',zorder=10,
                    bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)),clip_on=True)
            xyplotted.append((x,y))



