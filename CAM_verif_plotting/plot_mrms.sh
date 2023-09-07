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


DATE=$1

python /lfs/h2/emc/vpppg/save/logan.dawson/CAM_verif/plotting/mrms_plot.py $DATE 


exit



#ssh ldawson@emcrzdm "mkdir -p /home/people/emc/www/htdocs/users/Logan.Dawson/MEG/MRMS/${DATE}"
#scp /gpfs/dell2/ptmp/Logan.Dawson/MRMS_graphx/${DATE}/*.png ldawson@emcrzdm:/home/people/emc/www/htdocs/users/Logan.Dawson/MEG/MRMS/${DATE}
#     VTIME=${DATE:0:8}$(printf "%02d" $HR)



if [[ ${DATE:8:2} != "11" && ${DATE:8:2} != "23" ]]; then
   echo "exiting normally"
else
   echo "copying images to rzdm"
   bsub < /gpfs/dell2/emc/verification/save/Logan.Dawson/CAM_verif/plotting/ftp_mrms.sh
fi



analyses="refc refcprob50 refd refdprob40 retop retopprob30"
domains="conus ne se midatl glakes nc sc nw sw ca"

for analysis in $analyses; do

   # Make sure analysis directory exists
   ANALYSIS_DIR=/home/people/emc/www/htdocs/users/meg/hrefv3/analyses/${DATE:0:8}/${analysis}
 # ssh ldawson@emcrzdm.ncep.noaa.gov "mkdir -m 775 -p $ANALYSIS_DIR"

   for domain in $domains; do

      # Make sure domain directory exists
      DOMAIN_DIR=${ANALYSIS_DIR}/${domain}
      ssh ldawson@emcrzdm.ncep.noaa.gov "mkdir -m 775 -p $DOMAIN_DIR"

      # Copy images to directory
      scp /gpfs/dell2/ptmp/Logan.Dawson/MRMS_graphx/${DATE:0:8}/${analysis}_${domain}*.png ldawson@emcrzdm.ncep.noaa.gov:${DOMAIN_DIR}  

      # Copy files.php to directory
      scp /gpfs/dell2/emc/verification/save/Logan.Dawson/CAM_verif/plotting/files.php ldawson@emcrzdm.ncep.noaa.gov:${DOMAIN_DIR}
        
   done

done

exit


