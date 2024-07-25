import logging
from qaliboo import precomputed_functions
from qaliboo.pakman import PAKMAN
import numpy as np
from qaliboo.datasets import Dataset
from qaliboo.precomputed_functions import _PrecomputedFunction

logging.basicConfig(level=logging.NOTSET)
_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)

'''
ScaledQuery26 = Dataset(
    csv_file='scaledQuery26.csv',
    param_cols=['#vm', 'ram'],
    target_col='cost',
    time_col = 'time',
    Realtime_col = 'time',
    reduce_to_unique=False
)
'''
Query26 = Dataset(
    csv_file='query26_vm_ram.csv',
    param_cols=['#vm', 'ram'],
    target_col='cost',
    time_col = 'time',  # TODO change  the name of this variable
    Realtime_col='time',
    reduce_to_unique=False
)
#Query26 = _PrecomputedFunction(dataset=ScaledQuery26) 
Query26 = _PrecomputedFunction(dataset=Query26)

known_minimum = Query26.minimum
domain = Query26
objective_func = Query26
#objective_func_name = 'ScaledQuery26'
objective_func_name = None
n_initial_points = 3
n_iterations = 60
batch_size = 2
ub = 185000
nm = True

Baop = PAKMAN(n_initial_points=n_initial_points, 
           n_iterations=n_iterations, 
           batch_size=batch_size, 
           objective_func=objective_func,
           domain=objective_func,
           objective_func_name=objective_func_name,  
           ub=ub, 
           nm=nm,
           uniform_sample=True,
           save=True)

# 60 for LiGen (in teoria per 5)
# 36 for StereoMatch (in teoria per 250)
# 33 for StereoMatch10 (in teoria per 3)
#Baop.async_optimization(10, n_points_per_iteration, qua mettere il time proportion) # Cambia il time
Baop.sync_optimization()