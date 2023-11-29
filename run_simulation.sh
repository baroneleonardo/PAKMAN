#!/bin/bash

for q in {8..15}
do
    python main_test.py -p Rastrigin9 -i 11 -q $q -n 15 -nm True
done

