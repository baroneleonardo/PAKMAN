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
from examples import bayesian_optimization, auxiliary, synthetic_functions
from qaliboo import precomputed_functions, finite_domain
from qaliboo import simulated_annealing as SA
from qaliboo import sga_kg as sga
from qaliboo.machine_learning_models import ML_model

logging.basicConfig(level=logging.NOTSET)
_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)

class BayesianOptimization:
    def __init__(self, objectiv_func=None, known_minimum=None, domain=None):
        self._objective_func = objectiv_func 
        self._known_minimum = known_minimum
        self._domain = domain
    
    @property
    def objective_func(self):
        return self._objective_func
    @property
    def known_minimum(self):
        return self.known_minimum
    @property
    def domain(self):
        return self.domain
    
    def maximize(self, 
                 init_point,
                 n_iter,
                 q,
                 sa_cost = None,
                 acq = 'KG',
                 m_dom_disc = 20):
        
        initial_points_array = self.domain.sample_points_in_domain(init_point)
        initial_points_value = np.array([self.objective_func.evaluate(pt) for pt in initial_points_array])


        initial_points = [data_containers.SamplePoint(pt,
                                                      initial_points_value[num])
                          for num, pt in enumerate(initial_points_array)]
        initial_data = data_containers.HistoricalData(dim=self.objective_func.dim)
        initial_data.append_sample_points(initial_points)

        if acq == 'KG_ml' or acq == 'EI_ml':
            ml_model = ML_model(X_data=initial_points_array,
                                y_data=np.array([objective_func.evaluate_time(pt) for pt in initial_points_array]), 
                                X_ub=2.5) # Set this value if you are intrested in I(T(X) < X_ub)

        