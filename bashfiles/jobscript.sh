#!/bin/sh  
#BSUB -q gpua100
#BSUB -J NeuralCF
#BSUB -W 72:00
#BSUB -B
#BSUB -N

#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"

#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load cuda/11.8

source venv/bin/activate

python src/train/nfc_train.py
