#!/bin/bash

#SBATCH --partition=gpu        # Partition (job queue)

#SBATCH --requeue                 # Return job to the queue if preempted

#SBATCH --nodes=1                 # Number of nodes you require

#SBATCH --ntasks=1                # Total # of tasks across all nodes

#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)

#SBATCH --mem=64000                # Real memory (RAM) required (MB)

#SBATCH --gres=gpu:4

#SBATCH --time=3-00:00:00           # Total run time limit (D-HH:MM:SS)

#SBATCH --output=slurm/gpu/%j.out  # STDOUT output file

#SBATCH --error=slurm/gpu/%j.err   # STDERR output file (optional)

#SBATCH --exclude=cuda001

day=$(date +'%m/%d/%Y %R')
echo $@
echo ${day} $SLURM_JOBID "node_list" $SLURM_NODELIST $@  "\n" >> jobs.txt
module purge
module load intel/17.0.4
module load cudnn/7.0.3
module load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate cvtf
srun python3 $@
conda deactivate