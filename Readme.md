# Evolving counting in simple agents - code

## Contents
1. Code Organisation
2. Running the code
3. Branch Structure


## 1. Code Organisation 
```
BeeCommunication
└── src_analysis
└── src_evol
    ├── bin/
    │   └── main
    ├── build/
    │   ├── CountingAgent.o
    │   ├── CTRNN.o
    │   ├── main.o
    │   ├── random.o
    │   └── TSearch.o
    ├── include/
    │   ├── CountingAgent.h
    │   ├── CTRNN.h
    │   ├── random.h
    │   ├── TSearch.h
    │   └── VectorMatrix.h
    ├── src/
    │   ├── CountingAgent.cpp
    │   ├── CTRNN.cpp
    │   ├── main.cpp
    │   ├── random.cpp
    │   └── TSearch.cpp
    ├── Makefile
    └── runscrip.py
```

### Code Meaning

| File | Purpose |
| ---- | ------- | 
| TSearch | Evolves + searches the population for the best genotype in the population | 

## 2. Running the code
1. Navigate to the source directory (src_evol)
2. run: ```make```
3. run: ```python3 runscript.py x y z k bin/main``` 
where x is the start of the range to run, y is the end of the range to run
z is the index of the array and k is the index of the run


## The output

| File | Description |
| ---- | ----------- |
| seed_x | The seed of the run x |
| evol_x | Tracks each of the evolutions. Outputs the Generation, the best fitness and average fitness |
| best_gen_x | The genotype of the best individual in the population in run x | 
| best_ns_r_x | Save the phenotype of the best Reciever |
| best_ns_s_x | Save the phenotype of the best Signaler |


## 3. Branch Structure

{branch name} - {purpose}

reproduction - The original counting task (including extension)

referential counting task:
mode3-v1 - the single agent going to the food with a fixed input
simple - the simple referential communicaiton task with simple agents
mode3-v3 - the simple referential communication task with the full agents
mode3-v5 - th full referential communication task 
testing - used to perform the generalisation assessments on the four selected networks