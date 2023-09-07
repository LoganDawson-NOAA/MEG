#!/bin/bash
# Author: L.C. Dawson
#
#######################################################################
# Cron script to handle MRMS files for verification and HPSS          #
# Takes one argument to define which python script to run:            #
#    'copy' - copies MRMS files from dcom and unzip for verification  #
#    'archive' - zips and archives MRMS files to HPSS                 #
#######################################################################

set +x

source ~/.bashrc

set -x

DATE=$1

python /lfs/h2/emc/vpppg/save/${USER}/CAM_verif/plotting/ppf_plot.py $DATE 

# Transfer images to rzdm
#qsub ${SAVE_DIR}/plotting/ftp_ppf.sh

echo 'done'

exit
