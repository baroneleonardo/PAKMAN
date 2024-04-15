import logging
import argparse
from qaliboo import precomputed_functions
from qaliboo.parallel_maliboo import ParallelMaliboo as PM
import multiprocessing
logging.basicConfig(level=logging.NOTSET)
_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)

AVAILABLE_PROBLEMS = ['Query26','LiGen','StereoMatch','LiGenTot','ScaledLiGen','ScaledLiGenTot','ScaledStereoMatch','ScaledQuery26']

parser = argparse.ArgumentParser(prog='QALIBOO: Simplified finite domain q-KG',
                                 description='QALIBOO: Simplified finite domain q-KG',
                                 usage='Specify the selected problem and the other parameters.'
                                       ' Results are saved in the results/simplified_runs folder')
parser.add_argument('--problem', '-p', help='Selected dataset', choices=AVAILABLE_PROBLEMS, required=True)
parser.add_argument('--init', '-i', help='Number of initial points', type=int, default=7)
parser.add_argument('--iter', '-n', help='Number of iterations', type=int, default=9)
parser.add_argument('--points', '-q', help='Points per iteration (the `q` parameter)', type=int, default=7)
parser.add_argument('--sample_size', '-m', help='GP sample size (`M` parameter)', type=int, default=30)
parser.add_argument('--upper_bound', '-ub', help='Upper Bound (ML model)', type=float, default=None)
parser.add_argument('--lower_bound', '-lb', help='Lower Bound (ML model)', type=float, default=None)
parser.add_argument('--nascent_minima', '-nm', help='Nascent Minima term (ML model)', type=bool, default=False)
params = parser.parse_args()

objective_func_name = params.problem

objective_func = getattr(precomputed_functions, params.problem)
known_minimum = objective_func.minimum
domain = objective_func

n_initial_points = params.init
n_iterations = params.iter
n_points_per_iteration = params.points
m_domain_discretization_sample_size = params.sample_size
lb = params.lower_bound
ub = params.upper_bound
nm = params.nascent_minima

num_processors = multiprocessing.cpu_count()
print("Maximum number of available process", num_processors)

Baop = PM(n_initial_points=n_initial_points, 
           n_iterations=n_iterations, 
           batch_size=n_points_per_iteration, 
           m_domain_discretization=m_domain_discretization_sample_size,
           objective_func=objective_func,
           domain=objective_func, 
           lb=lb, 
           ub=ub, 
           nm=nm,
           uniform_sample=True,
           save=True)

# 60 for LiGen (in teoria per 5)
# 36 for StereoMatch (in teoria per 250)
Baop.async_optimization(60, n_points_per_iteration) # Cambia il time
#Baop.sync_optimization()
