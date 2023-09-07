#!/bin/bash
#BSUB -J ftp_mrms
#BSUB -o /gpfs/dell2/ptmp/Logan.Dawson/cron.out/ftp_mrms.%J.out
#BSUB -e /gpfs/dell2/ptmp/Logan.Dawson/cron.out/ftp_mrms.%J.out
#BSUB -n 1
#BSUB -W 00:10
#BSUB -P HRW-T2O
#BSUB -q dev_transfer
#BSUB -R "rusage[mem=1000]"
#BSUB -R "affinity[core]"
#
set -x

now=`date -u +%Y%m%d%H`

if [[ ${now:8:2} == "11" || ${now:8:2} == "12" || ${now:8:2} == "23" ]]; then
   DATE=$now
elif [[ ${now:8:2} == "00" ]]; then
   DATE=`$NDATE -24 $now | cut -c 1-10`
else
   echo "error setting correct date"
   exit
fi

echo "copying ${DATE:0:8} images to rzdm"

analyses="refc refcprob50 refd refdprob40 retop retopprob30"
domains="conus ne se midatl glakes nc sc nw sw ca"

for analysis in $analyses; do

   # Make sure analysis directory exists
   ANALYSIS_DIR=/home/people/emc/www/htdocs/users/meg/hrefv3/analyses/${DATE:0:8}/${analysis}
   ssh ldawson@emcrzdm.ncep.noaa.gov "mkdir -m 775 -p $ANALYSIS_DIR"

   # Copy index.html to analysis directory
   scp /gpfs/dell2/emc/verification/save/Logan.Dawson/CAM_verif/plotting/index.html ldawson@emcrzdm.ncep.noaa.gov:${ANALYSIS_DIR}

   for domain in $domains; do

      # Make sure domain directory exists
      DOMAIN_DIR=${ANALYSIS_DIR}/${domain}
      ssh ldawson@emcrzdm.ncep.noaa.gov "mkdir -m 775 -p $DOMAIN_DIR"

      # Copy images to domain directory
      scp /gpfs/dell2/ptmp/Logan.Dawson/MRMS_graphx/${DATE:0:8}/${analysis}_${domain}*.png ldawson@emcrzdm.ncep.noaa.gov:${DOMAIN_DIR}

      # Copy files.php to domain directory
      scp /gpfs/dell2/emc/verification/save/Logan.Dawson/CAM_verif/plotting/files.php ldawson@emcrzdm.ncep.noaa.gov:${DOMAIN_DIR}

   done

done

exit
