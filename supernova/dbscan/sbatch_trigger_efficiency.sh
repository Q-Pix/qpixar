#!/usr/bin/env bash

#SBATCH -J ve_cc_dbscan_qpix   # A single job name for the array
#SBATCH -n 1                   # Number of cores
#SBATCH -N 1                   # All cores on one machine
#SBATCH -p guenette            # Partition
#SBATCH --mem 4000             # Memory request (Mb)
#SBATCH -t 0-48:00             # Maximum execution time (D-HH:MM)
#SBATCH -o /n/holyscratch01/guenette_lab/Users/jh/supernova/dbscan/log/%A_%a.out        # Standard output
#SBATCH -e /n/holyscratch01/guenette_lab/Users/jh/supernova/dbscan/log/%A_%a.err        # Standard error

offset=0

# SCRATCH_DIR="/n/holyscratch01/guenette_lab/Users/jh/supernova/dbscan/root"
SCRATCH_DIR="/scratch/`whoami`"
STORE_DIR="/n/holystore01/LABS/guenette_lab/Lab/data/q-pix/supernova"
OUTPUT_DIR="${STORE_DIR}/dbscan"
LOG_DIR=${SCRATCH_DIR}
# OUTPUT_DIR="${STORE_DIR}/signal"

if [ ! -d "${SCRATCH_DIR}" ]; then
  mkdir -p ${SCRATCH_DIR}
fi

LOG_PREFIX="${SLURM_ARRAY_JOB_ID}"_"${SLURM_ARRAY_TASK_ID}"
LOG_PATH=${LOG_DIR}/${LOG_PREFIX}

# PY_DBSCAN=/n/home02/jh/repos/qpixar/supernova/dbscan/dbscan.py
# PY_DBSCAN=/n/home02/jh/repos/qpixar/supernova/dbscan/dbscan2.py
# PY_DBSCAN=/n/home02/jh/repos/qpixar/supernova/dbscan/dbscan3.py
# PY_DBSCAN=/n/home02/jh/repos/qpixar/supernova/dbscan/dbscan_euclidean.py
# PY_DBSCAN=/n/home02/jh/repos/qpixar/supernova/dbscan/multi_stage_dbscan.py
# PY_DBSCAN=/n/home02/jh/repos/qpixar/supernova/dbscan/trigger_efficiency.py
PY_DBSCAN=/n/home02/jh/repos/qpixar/supernova/dbscan/dbscan_chebyshev_trigger_efficiency.py

index=$(echo `expr ${SLURM_ARRAY_TASK_ID} + $offset`)
index_lz=$(printf "%06d" "$index")

signal_idx=0
background_idx=0

signal_idx=$index
background_idx=$index

reaction="ve_cc"
# reaction="ve_es"

first=0
last=5000
step=1

# increment=$(echo `expr $last - $first + $step`)
# increment=$(echo `expr ${SLURM_ARRAY_TASK_ID} \* $increment`)
# 
# first=$(echo `expr $first + $increment`)
# last=$(echo `expr $last + $increment`)

events=($(seq $first $step $last))

# echo ${events[*]}

# root_file_name=multi_stage_dbscan_"$reaction"_"${SLURM_ARRAY_JOB_ID}"_"$index_lz".root
# root_file_name=trigger_efficiency_"$reaction"_"${SLURM_ARRAY_JOB_ID}"_"$index_lz".root
root_file_name=dbscan_chebyshev_trigger_efficiency_"$reaction"_"${SLURM_ARRAY_JOB_ID}"_"$index_lz".root
root_file=${SCRATCH_DIR}/$root_file_name
output_file_path=${OUTPUT_DIR}/$index_lz

function main() {

  if [ ! -d "$output_file_path" ]; then
    mkdir -p $output_file_path
  fi

  date; sleep 2
  time python -u $PY_DBSCAN $signal_idx $background_idx --reaction $reaction --events $(echo ${events[*]}) --output $root_file
  date; sleep 2
  mv $root_file $output_file_path
  date; sleep 2

}

function signal_handler() {
  echo "Catching signal"
  cd $SLURM_SUBMIT_DIR
  mkdir -p $SLURM_ARRAY_JOB_ID
  touch ${SLURM_ARRAY_JOB_ID}/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_caught_signal
  # cp -R $TMPDIR/* $SLURM_JOB_ID
  cp ${LOG_PATH}.{out,err} ${SLURM_ARRAY_JOB_ID}
  exit
}  

trap signal_handler USR1
trap signal_handler TERM

main 1> ${LOG_PATH}.out 2> ${LOG_PATH}.err &
wait

