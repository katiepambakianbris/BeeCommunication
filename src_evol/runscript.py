###############################################
# runs the program many times 
# and reports how long it takes to complete
###############################################

# running command
# python3 runscript.py 0 100 bin/main

import os
import sys

fromR = int(sys.argv[1]) 
toR = int(sys.argv[2])
array_index = int(sys.argv[3])
program = sys.argv[4]

currentpath = os.getcwd()

for k in range(fromR,toR):
    print(k)
    # time main trail_number slurm_array_index
    os.system('time ./'+program+" "+str(k) + " "+ str(array_index))
