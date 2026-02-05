###############################################
# runs the program many times 
# and reports how long it takes to complete
###############################################

# running command
# python3 runscript.py 0 100 bin/main

import os
import sys

# the range of trials to run
fromR = int(sys.argv[1]) 
toR = int(sys.argv[2])
# the index in the slurm array
array_index = int(sys.argv[3])
# the slurm job id
slurm_job_id = int(sys.argv[4])
# the binary to execute
program = sys.argv[5]

currentpath = os.getcwd()

for k in range(fromR,toR):
    print(k)
    # time main trail_number slurm_array_index slurm_job_index
    os.system('time ./'+program+" "+str(k) + " "+ str(array_index) +" " + str(slurm_job_id))
