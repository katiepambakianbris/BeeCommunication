#!/bin/bash

#SBATCH --job-name=initial_bee_100_trials
#SBATCH --output=initial_bee_100_trials.out    # Standard output
#SBATCH --error=initial_bee_100_trials.err     # Error log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=1000M
#SBATCH --account=COSC029884
#SBATCH --array=0-19

module add languages/python/3.12.3
 
SRC_DIR="../BeeCommunication/src_evol"

# running the runscript.py 
python $SRC_DIR/runscript.py 0 5 $SLURM_ARRAY_TASK_ID $SRC_DIR/bin/main 