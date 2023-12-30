#!/bin/bash

############## Q = 4 #####################

python main_test.py -p Schwefel5 -i 14 -n 35 -q 4 

python main_test.py -p Schwefel5 -i 14 -n 35 -q 5 

python main_test.py -p Schwefel5 -i 14 -n 35 -q 4 -lb 380

python main_test.py -p Schwefel5 -i 14 -n 35 -q 5 -lb 380
