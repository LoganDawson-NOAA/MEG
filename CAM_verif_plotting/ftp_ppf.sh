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

NOSCRUB_DIR=/lfs/h2/emc/vpppg/noscrub/${USER}/CAM_verif/surrogate_severe/point2grid
RZDM_DIR=/home/people/emc/www/htdocs/users/verification/regional/cam/ops/surrogate_severe/maps/point2grid/${DATE}


# Make sure rzdm directory exists
ssh ldawson@emcrzdm.ncep.noaa.gov "mkdir -m 775 -p ${RZDM_DIR}/${DATE:0:6}"

# Copy index.html to analysis directory
scp ${NOSCRUB_DIR}/${DATE:0:6}/*.png ldawson@emcrzdm:${RZDM_DIR}/${DATE:0:6}


exit


directories="pcp_combine sspf"

for directory in $directories; do

   RZDM_DIR=${RZDM_HEAD}/${directory}/${DATE}
   ssh ldawson@emcrzdm.ncep.noaa.gov "mkdir -m 775 -p $RZDM_DIR"

   scp ${NOSCRUB_DIR}/${directory}/${DATE}/*.png ldawson@emcrzdm.ncep.noaa.gov:${RZDM_DIR}

done

exit
