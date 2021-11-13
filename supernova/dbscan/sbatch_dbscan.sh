#!/usr/bin/env bash

#SBATCH -J dbscan_ve_cc_qpix   # A single job name for the array
#SBATCH -n 1                   # Number of cores
#SBATCH -N 1                   # All cores on one machine
#SBATCH -p guenette            # Partition
#SBATCH --mem 1500             # Memory request (Mb)
#SBATCH -t 0-05:00             # Maximum execution time (D-HH:MM)
#SBATCH -o /n/holyscratch01/guenette_lab/Users/jh/supernova/dbscan/log/%A_%a.out        # Standard output
#SBATCH -e /n/holyscratch01/guenette_lab/Users/jh/supernova/dbscan/log/%A_%a.err        # Standard error

offset=0

SCRATCH_DIR="/n/holyscratch01/guenette_lab/Users/jh/supernova/dbscan/root"
STORE_DIR="/n/holystore01/LABS/guenette_lab/Lab/data/q-pix/supernova"
OUTPUT_DIR="${STORE_DIR}/dbscan"

PY_DBSCAN=/n/home02/jh/repos/qpixar/supernova/dbscan/dbscan.py

index=$(echo `expr ${SLURM_ARRAY_TASK_ID} + $offset`)
index_lz=$(printf "%06d" "$index")

signal_idx=0
background_idx=0

signal_idx=$index
background_idx=$index

reaction="ve_cc"
# reaction="ve_es"

first=0
last=50
step=1

# increment=$(echo `expr $last - $first + $step`)
# increment=$(echo `expr ${SLURM_ARRAY_TASK_ID} \* $increment`)
# 
# first=$(echo `expr $first + $increment`)
# last=$(echo `expr $last + $increment`)

events=($(seq $first $step $last))

# echo ${events[*]}

root_file_name=dbscan_"$reaction"_"$index_lz".root
root_file=${SCRATCH_DIR}/$root_file_name
output_file_path=${OUTPUT_DIR}/$index_lz

if [ ! -d "$output_file_path" ]; then
    mkdir -p $output_file_path
fi

date; sleep 2
time python $PY_DBSCAN $signal_idx $background_idx --reaction $reaction --events $(echo ${events[*]}) --output $root_file
date; sleep 2
mv $root_file $output_file_path
date; sleep 2

