#!/bin/bash

for q in {15..20}
do
    python MAIN.py -p ScaledLiGenTot -i 11 -q $q -n 15 -ub 2.7 
done

for q in {10..15}
do
    python main.py -p ScaledLiGenTot -i 11 -q $q -n 15 
done
