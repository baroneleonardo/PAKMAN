
for i in {1..9}
do
    python main_testfunction.py -p Hartmann6 -i 3 -q 6 -n 30 -lb 1.0 
    python main_testfunction.py -p Rastrigin5 -i 3 -q 5 -n 30 -lb 1.0
    python main_testfunction.py -p Ackley5 -i 3 -q 6 -n 30 -lb 1.5
    python main_testfunction.py -p Schwefel5 -i 3 -q 4 -n 30 -lb 930
    python main_testfunction.py -p Hartmann6 -i 3 -q 6 -n 30 -lb 1.0 -nm True
    python main_testfunction.py -p Rastrigin5 -i 3 -q 5 -n 30 -lb 1.0 -nm True
    python main_testfunction.py -p Ackley5 -i 3 -q 6 -n 30 -lb 1.5 -nm True
    python main_testfunction.py -p Schwefel5 -i 3 -q 4 -n 30 -lb 960 -nm True
    python main_testfunction.py -p Hartmann6 -i 3 -q 6 -n 30 -nm True
    python main_testfunction.py -p Rastrigin5 -i 3 -q 5 -n 30 -nm True
    python main_testfunction.py -p Ackley5 -i 3 -q 6 -n 30 -nm True
    python main_testfunction.py -p Schwefel5 -i 3 -q 4 -n 30 -nm True
    
done

for i in {1..9}
do
    python main_testfunction.py -p Hartmann6 -i 3 -q 6 -n 30 -lb 1.6 
    python main_testfunction.py -p Rastrigin5 -i 3 -q 5 -n 30 -lb 1.6
    python main_testfunction.py -p Ackley5 -i 3 -q 6 -n 30 -lb 2.2
    python main_testfunction.py -p Schwefel5 -i 3 -q 4 -n 30 -lb 980
    python main_testfunction.py -p Hartmann6 -i 3 -q 6 -n 30 -lb 1.6 -nm True
    python main_testfunction.py -p Rastrigin5 -i 3 -q 5 -n 30 -lb 1.6 -nm True
    python main_testfunction.py -p Ackley5 -i 3 -q 6 -n 30 -lb 2.2 -nm True
    python main_testfunction.py -p Schwefel5 -i 3 -q 4 -n 30 -lb 980 -nm True
    
done


#for i in {1..9}
#do
#    python main_testfunction.py -p Hartmann6 -i 3 -q 6 -n 30 -lb 1.4 
#    python main_testfunction.py -p Rastrigin5 -i 3 -q 5 -n 30 -lb 1.4
#    python main_testfunction.py -p Ackley5 -i 3 -q 6 -n 30 -lb 2
#    python main_testfunction.py -p Schwefel5 -i 3 -q 4 -n 30 -lb 960
#    python main_testfunction.py -p Hartmann6 -i 3 -q 6 -n 30 -lb 1.4 -nm True
#    python main_testfunction.py -p Rastrigin5 -i 3 -q 5 -n 30 -lb 1.4 -nm True
#    python main_testfunction.py -p Ackley5 -i 3 -q 6 -n 30 -lb 2 -nm True
#    python main_testfunction.py -p Schwefel5 -i 3 -q 4 -n 30 -lb 960 -nm True
#    
#done