#!/bin/bash
#SBATCH --job-name=bee_hard_mode_only
#SBATCH --output=results/slurm/%x/%A/%x-%A-run-%a.out    # Standard output
#SBATCH --error=results/slurm//%x/%A/%x-%A-run-%a.err     # Error log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem=1000M
#SBATCH --account=COSC029884
#SBATCH --array=0-19

# run this script from the root BeeCommunication directory
# sbatch scripts/bee-script.sh

module add languages/python/3.12.3
 
SRC_DIR="src_evol"

# make the files 

cd $SRC_DIR
make 
cd ..

# running the runscript.py 
python $SRC_DIR/runscript.py 0 1 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_JOB_ID $SRC_DIR/bin/main 