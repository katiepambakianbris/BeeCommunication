# Evolving counting in simple agents - code

## Contents
1. Code Organisation
2. Running the code


## 1. Code Organisation 

BeeCommunication
└── src_analysis
└── src_evol
    └── bin/
        └── main
    └── build/
        ├── CountingAgent.o
        ├── CTRNN.o
        ├── main.o
        ├── random.o
        └── TSearch.o
    └── include/
        ├── CountingAgent.h
        ├── CTRNN.h
        ├── random.h
        ├── TSearch.h
        └── VectorMatrix.h
    └── src/
        ├── CountingAgent.cpp
        ├── CTRNN.cpp
        ├── main.cpp
        ├── random.cpp
        └── TSearch.cpp
    └── Makefile
    └── runscript.py

### Code Meaning

| File | Purpose |
| ---- | ------- | 
| TSearch | Evolves + searches the population for the best genotype in the population | 

## Running the Code


## The output

| File | Description |
| ---- | ----------- |
| seed_x | The seed of the run x |
| evol_x | |
| best_gen_x | The genotype of the best individual in the population in run x | 
| best_ns_r_x | Save the phenotype of the best Reciever |
| best_ns_s_x | Save the phenotype of the best Signaler |
