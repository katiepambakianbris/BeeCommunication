#!/bin/bash

g++ -std=c++17 test.cpp -o test
./test
python3 plot.py
