###############################################
# runs the program many times 
# and reports how long it takes to complete
###############################################

# running command
# python3 runscript.py 0 1 bin/main

import os
import sys

fromR = int(sys.argv[1])
toR = int(sys.argv[2])
program = sys.argv[3]

currentpath = os.getcwd()

for k in range(fromR,toR):
    print(k)
    os.system('time ./'+program+" "+str(k))
