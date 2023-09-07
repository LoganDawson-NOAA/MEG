#!/bin/ksh
#BSUB -J ec_htar
#BSUB -o ec_htar.%J.out
#BSUB -e ec_htar.%J.out
#BSUB -n 1
#BSUB -W 01:00
#BSUB -P HRRR-T2O
#BSUB -q dev_transfer
#BSUB -R "rusage[mem=1000]"
#BSUB -R "affinity[core]"

source ~/.bashrc

#==============================================  BEGIN CHANGES  ================================================

CYCLE=2021102312

FHR_START=0
FHR_END=84
FHR_INC=6

#===============================================  END CHANGES  =================================================

if [ $USER == "Alicia.Bentley" ]; then
   PTMP_LOC="p2"
elif [ $USER == "Logan.Dawson" ]; then
   PTMP_LOC="d2"
elif [ $USER == "Tracey.Dorian" ]; then
   PTMP_LOC="d3"
elif [ $USER == "Geoffrey.Manikin" ]; then
   PTMP_LOC="d1"
else
   PTMP_LOC="p1"
fi

RETRO_DIR="/gpfs/dell2/emc/verification/noscrub/Logan.Dawson/MEG/data"

mkdir -p $RETRO_DIR

cd $RETRO_DIR

/bin/rm -rf htar_ecanl_done ${CYCLE}_valids.txt

yyyy=`echo $CYCLE | cut -c 1-4`
yyyymm=`echo $CYCLE | cut -c 1-6`
yyyymmdd=`echo $CYCLE | cut -c 1-8`
mm=`echo $CYCLE | cut -c 5-6`
dd=`echo $CYCLE | cut -c 7-8`
hh=`echo $CYCLE | cut -c 9-10`

file="${CYCLE}_valids.txt"
if [[ -e ${RETRO_DIR}/${file} ]] ; then
   echo ""
else
   python /gpfs/dell2/emc/verification/save/Logan.Dawson/MEG/valids.py $CYCLE $FHR_START $FHR_END $FHR_INC
fi

#===============================================  GET ANALYSES  =================================================

FHR=$FHR_START
while IFS= read -r line ; do
   VALID="`echo $line`"
   YYYY=`echo $VALID | cut -c 1-4`
   YYYYMM=`echo $VALID | cut -c 1-6`
   YYYYMMDD=`echo $VALID | cut -c 1-8`
   MM=`echo $VALID | cut -c 5-6`
   DD=`echo $VALID | cut -c 7-8`
   HH=`echo $VALID | cut -c 9-10`

   let "TEMPFHR=FHR+1000"
   FHR3="`echo $TEMPFHR | cut -c 2-`"


##### ECMWF
   EC_ARCHIVE=/NCEPPROD/hpssprod/runhistory/rh${yyyy}/${yyyymm}/${yyyymmdd}/dcom_prod_${yyyymmdd}.tar

   # make temporary directory to download into
   mkdir -p $RETRO_DIR/ecmwf.${yyyymmdd}
   cd $RETRO_DIR/ecmwf.${yyyymmdd}

   if [[ ${CYCLE} == ${VALID} ]] ; then
      echo "Extracting "${VALID}" EC analysis"
      htar -xvf $EC_ARCHIVE ./wgrbbul/ecmwf/U1D${mm}${dd}${hh}00${MM}${DD}${HH}011
      mv ./wgrbbul/ecmwf/U1D${mm}${dd}${hh}00${MM}${DD}${HH}011 ${RETRO_DIR}/ecmwf.${yyyymmdd}/ec.t${hh}z.0p25.f${FHR3}.grb
   else
      echo "Extracting "${CYCLE}" EC forecast"
      htar -xvf $EC_ARCHIVE ./wgrbbul/ecmwf/U1D${mm}${dd}${hh}00${MM}${DD}${HH}001
      mv ./wgrbbul/ecmwf/U1D${mm}${dd}${hh}00${MM}${DD}${HH}001 ${RETRO_DIR}/ecmwf.${yyyymmdd}/ec.t${hh}z.0p25.f${FHR3}.grb
   fi

   let "FHR=FHR+FHR_INC"

done <"$file"
#mv ./wgrbbul/ecmwf/* ${RETRO_DIR}/ec.${yyyymmdd}
#==============================================================================================================

cd $RETRO_DIR
#/bin/rm -fR $RETRO_DIR/ecanl.${CYCLE} wgrbbul
/bin/rm -fR $RETRO_DIR/ecmwf.${yyyymmdd}/wgrbbul

touch htar_ecanl_done



exit
