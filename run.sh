#!/bin/bash

#python main_time.py -p ScaledLiGenTot -i 40 -q 10 -n 100 -ub 2.1
#python main_time.py -p ScaledLiGenTot -i 40 -q 10 -n 100 -ub 2.1
#python main_time.py -p ScaledLiGenTot -i 40 -q 10 -n 100 -ub 2.1
#python main_time.py -p ScaledLiGenTot -i 40 -q 10 -n 100 -ub 2.1

#python main_time.py -p ScaledLiGenTot -i 10 -q 7 -n 100 -ub 2.1  # Questo per execution time

#python asyn_test.py -p ScaledLiGenTot -i 40 -q 10 -n 200 -ub 2.1
#python asyn_test.py -p ScaledLiGenTot -i 40 -q 10 -n 200 -ub 2.1
#python asyn_test.py -p ScaledLiGenTot -i 40 -q 10 -n 200 -ub 2.1
#python asyn_test.py -p ScaledLiGenTot -i 40 -q 10 -n 200 -ub 2.1

# Asyncronous symulations
#for i in {1..5} 
#do
#    python main_time.py -p ScaledStereoMatch -i 5 -q 4 -n 30 -ub 20000
#    python main_time.py -p ScaledStereoMatch -i 5 -q 4 -n 30 -nm True
#    python main_time.py -p ScaledStereoMatch -i 5 -q 4 -n 30 -ub 20000 -nm True
# python main_time.py -p      
 
#done
#    python main_test.py -p XGBoostBinary -q 4 -n 30 -lb 0.33
#    python main_test.py -p XGBoostRegressor -q 4 -n 30 -lb 0.5


# Syncronous simulations
#python asyn_test.py -p ScaledStereoMatch -i 5 -q 4 -n 1000 -ub 20000
# python asyn_test.py -p ScaledStereoMatch10 -i 5 -q 4 -n 1000 -ub 155 -nm True
#    python main_time.py -p ScaledStereoMatch10 -i 3 -q 4 -n 30 -ub 155 -nm True<
for i in {1..5}
do
    echo -e "\e[31mNumber of iteration: $i\e[0m"
    python asyn_test.py -p ScaledQuery26 -i 3 -q 2 -n 50 -ub 195000
    python asyn_test.py -p ScaledQuery26 -i 3 -q 2 -n 50 -ub 205000 
    python asyn_test.py -p ScaledQuery26 -i 3 -q 2 -n 50 -ub 225000
    python asyn_test.py -p ScaledQuery26 -i 3 -q 2 -n 50 -ub 185000
    python asyn_test.py -p ScaledQuery26 -i 3 -q 2 -n 50 -ub 195000 -nm True
    python asyn_test.py -p ScaledQuery26 -i 3 -q 2 -n 50 -ub 205000 -nm True
    python asyn_test.py -p ScaledQuery26 -i 3 -q 2 -n 50 -ub 225000 -nm True
    python asyn_test.py -p ScaledQuery26 -i 3 -q 2 -n 50 -ub 185000 -nm True
     
done
#constraints = 1000 * np.array((185, 195, 205, 215, 225))

#for i in {1..4}
#do
#    python main_time.py -p Query26 -i 3 -q 1 -n 30 -ub 215000
#done

#python main_testfunction.py -p Schwefel5 -i 3 -q 6 -n 30 -lb 960


#for i in {1..9}
#do
#    python main_testfunction.py -p Hartmann6 -i 3 -q 6 -n 30 -lb 1.4 
#    python main_testfunction.py -p Rastrigin5 -i 3 -q 5 -n 30 -lb 1.4
#    python main_testfunction.py -p Ackley5 -i 3 -q 6 -n 30 -lb 2
#    python main_testfunction.py -p Schwefel5 -i 3 -q 4 -n 30 -lb 960
#done