#!/bin/bash

for q in {10..15}
do
    python MAIN4.py -p ScaledLiGen -i 11 -q $q -n 15 -nm True -ub 2.7
done

