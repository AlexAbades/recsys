#!/bin/sh  
#BSUB -q gpua100
#BSUB -J NeuralCF-5
#BSUB -W 72:00
#BSUB -B
#BSUB -N

#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"

#BSUB -o src/logs/%J.out
#BSUB -e src/logs/%J.err

module load python3/3.10.12
module load cuda/11.8

source /zhome/c0/a/164613/Desktop/recsys/venv/bin/activate
export PYTHONPATH=/zhome/c0/a/164613/Desktop/recsys:$PYTHONPATH

python src/train/NCF.py --config configs/nfc/YELP/yelp1.yaml
