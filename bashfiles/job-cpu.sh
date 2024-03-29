#!/bin/sh  
#BSUB -q hpc
#BSUB -J Yelp-Transormation
#BSUB -W 72:00

#BSUB -R "span[hosts=1]"
#BSUB -n 6
#BSUB -R "rusage[mem=8GB]"

#BSUB -o src/logs/%J.out
#BSUB -e src/logs/%J.err

#BSUB -B
#BSUB -N

module load python3/3.10.12
module load cuda/11.8

source /zhome/c0/a/164613/Desktop/recsys/venv/bin/activate
export PYTHONPATH=/zhome/c0/a/164613/Desktop/recsys:$PYTHONPATH

python src/data/yelp_raw_processed.py
