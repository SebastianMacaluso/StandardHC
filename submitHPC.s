#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:59:00
#SBATCH --mem=32GB
#SBATCH --job-name=ClusteringAlgorithms
#SBATCH --mail-type=END
#SBATCH --mail-user=sm4511@nyu.edu
#SBATCH --output=logs/slurm_%${SLURM_ARRAY_TASK_ID}_%j.out

module purge

## executable
##SRCDIR=$HOME/ReclusterTreeAlgorithms/scripts

RUNDIR=$SCRATCH/TreeAlgorithms/runs/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR

##cd $SLURM_SUBMIT_DIR
##cp my_input_params.inp $RUNDIR

##cd $RUNDIR
##module load fftw/intel/3.3.5

cd $HOME/TreeAlgorithms/scripts/


## GREEDY
##python jetClustering.py --greedyScan=True --N_jets=500 --id=${SLURM_ARRAY_TASK_ID}

##Beam Search
##python jetClustering.py --greedyScan=True --BSScan=True --N_jets=2 --id=${SLURM_ARRAY_TASK_ID} --data_dir="/home/sm4511/TreeAlgorithms/data/truth/"




##python jetClustering.py --greedyScan=True --BSScan=True --KtAntiktCAscan=True --N_jets=100 --id=${SLURM_ARRAY_TASK_ID} --jetType=QCDjets
##python jetClustering.py --KtAntiktCAscan=True --N_jets=100 --id=${SLURM_ARRAY_TASK_ID} --jetType=Wjets

python jetClustering.py --greedyScan=True --BSScan=True --KtAntiktCAscan=True --N_jets=100 --id=${SLURM_ARRAY_TASK_ID} --jetType=Wjets

##To test:


##python jetClustering.py --greedyScan=True --N_jets=100 --id=${SLURM_ARRAY_TASK_ID} --jetType=QCDjets


##python jetClustering.py --greedyScan=True --BSScan=True --KtAntiktCAscan=True --N_jets=2 --id=${SLURM_ARRAY_TASK_ID} --jetType=QCDjets

##Notes:
##jetType= QCDjets , Wjets or Topjets

## to submit(for 2 jobs): sbatch --array 0-2 submitHPC.s

## to submit 50000 jets (for 2 jobs): sbatch --array 0-499 submitHPC.s

