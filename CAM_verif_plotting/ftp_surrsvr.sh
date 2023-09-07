#!/bin/bash
#BSUB -J ftp_surr_svr
#BSUB -o /gpfs/dell2/ptmp/Logan.Dawson/cron.out/ftp_svr.%J.out
#BSUB -e /gpfs/dell2/ptmp/Logan.Dawson/cron.out/ftp_svr.%J.out
#BSUB -n 1
#BSUB -W 00:10
#BSUB -P HRW-T2O
#BSUB -q dev_transfer
#BSUB -R "rusage[mem=1000]"
#BSUB -R "affinity[core]"
#
set -x

DATE=$ACCUM_END

echo "copying ${DATE} images to rzdm"

NOSCRUB_DIR=/gpfs/dell2/emc/verification/noscrub/Logan.Dawson/CAM_verif/surrogate_severe
RZDM_HEAD=/home/people/emc/www/htdocs/users/verification/regional/cam/ops/surrogate_severe/maps

directories="pcp_combine sspf"

for directory in $directories; do

   # Make sure rzdm directory exists
   RZDM_DIR=${RZDM_HEAD}/${directory}/${DATE}
   ssh ldawson@emcrzdm.ncep.noaa.gov "mkdir -m 775 -p $RZDM_DIR"

   # Copy index.html to analysis directory
   scp ${NOSCRUB_DIR}/${directory}/${DATE}/*.png ldawson@emcrzdm.ncep.noaa.gov:${RZDM_DIR}

done

exit
