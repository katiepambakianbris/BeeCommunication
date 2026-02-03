#!/bin/bash
 
#SBATCH --job-name=test_job
#SBATCH --output=test-job-%A-run-%a.out    # Standard output
#SBATCH --error=test-job-%A-run-%a.err     # Error log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=1000M
#SBATCH --account=COSC029884
#SBATCH --array=0-4
 
module add languages/python/3.12.3
 
python test/test.py $SLURM_ARRAY_TASK_ID


