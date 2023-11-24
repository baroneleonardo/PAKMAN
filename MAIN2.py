import datetime
import logging
import time
import json
import argparse

import numpy as np
import numpy.linalg

from moe.optimal_learning.python import data_containers
from moe.optimal_learning.python.cpp_wrappers import log_likelihood_mcmc, optimization as cpp_optimization, knowledge_gradient
from moe.optimal_learning.python.python_version import optimization as py_optimization
from moe.optimal_learning.python import default_priors
from moe.optimal_learning.python import random_features
from moe.optimal_learning.python.cpp_wrappers import knowledge_gradient_mcmc as KG
from moe.optimal_learning.python.cpp_wrappers import expected_improvement_mcmc as EI
from examples import bayesian_optimization, auxiliary, synthetic_functions
from qaliboo import precomputed_functions, finite_domain
from qaliboo import simulated_annealing as SA
from qaliboo import sga_kg as sga
from qaliboo.machine_learning_models import ML_model
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.NOTSET)
_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)

###########################
# Constants
###########################
N_RANDOM_WALKERS = 1 #2 ** 4
AVAILABLE_PROBLEMS = [
    # Toy problems:
    'ParabolicMinAtOrigin',
    'ParabolicMinAtTwoAndThree',
    # Benchmark functions:
    'Hartmann3',
    'Branin',
    'Ackley5',
    'Levy4',  # This function implementation is probably wrong
    # Data sets
    'Query26',
    'LiGen',
    'StereoMatch',
    'LiGenTot',
    'ScaledLiGen',
    'ScaledLiGenTot'
]

###########################
# Script parameters
###########################
parser = argparse.ArgumentParser(prog='QALIBOO: Simplified finite domain q-KG',
                                 description='QALIBOO: Simplified finite domain q-KG',
                                 usage='Specify the selected problem and the other parameters.'
                                       ' Results are saved in the results/simplified_runs folder')
parser.add_argument('--problem', '-p', help='Selected dataset', choices=AVAILABLE_PROBLEMS, required=True)
parser.add_argument('--init', '-i', help='Number of initial points', type=int, default=7)
parser.add_argument('--iter', '-n', help='Number of iterations', type=int, default=9)
parser.add_argument('--points', '-q', help='Points per iteration (the `q` parameter)', type=int, default=7)
parser.add_argument('--sample_size', '-m', help='GP sample size (`M` parameter)', type=int, default=30)
parser.add_argument('--queue_size', '-u', help='Dimension of the Queue', type=int, default=0)
params = parser.parse_args()

objective_func_name = params.problem

if objective_func_name == 'ParabolicMinAtOrigin':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([0.0, 0.0])
    domain = finite_domain.CPPFiniteDomain.Grid(np.arange(-5, 5, 0.1),
                                                np.arange(-5, 5, 0.1))
elif objective_func_name == 'ParabolicMinAtTwoAndThree':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([2.0, 3.0])
    domain = finite_domain.CPPFiniteDomain.Grid(np.arange(-5, 5, 0.1),
                                                np.arange(-5, 5, 0.1))
elif objective_func_name == 'Hartmann3':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([0.114614, 0.555649, 0.852547])
    domain = finite_domain.CPPFiniteDomain.Grid(np.arange(0, 1, 0.01),
                                                np.arange(0, 1, 0.01),
                                                np.arange(0, 1, 0.01))
elif objective_func_name == 'Branin':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([3.14, 2.28])
    domain = finite_domain.CPPFiniteDomain.Grid(np.arange(0, 15, 0.01),
                                                np.arange(-5, 15, 0.01))
elif objective_func_name=='Ackley5':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    domain = finite_domain.CPPFiniteDomain.Grid(np.arange(-1,1,0.1),np.arange(-1,1,0.1),np.arange(-1,1,0.1),np.arange(-1,1,0.1),np.arange(-1,1,0.1))

elif objective_func_name == 'Levy4':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([1.0, 1.0, 1.0, 1.0])
    domain = finite_domain.CPPFiniteDomain.Grid(np.arange(-1, 2, 0.1),np.arange(-1, 2, 0.1),np.arange(-1, 2, 0.1),np.arange(-1, 2, 0.1))
else:
    objective_func = getattr(precomputed_functions, params.problem)
    known_minimum = objective_func.minimum
    domain = objective_func

n_initial_points = params.init
n_iterations = params.iter
n_points_per_iteration = params.points
m_domain_discretization_sample_size = params.sample_size

