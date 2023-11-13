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


class UtilityFunction(object):
    def __init__(self, 
                 acq, 
                 gaussian_process_mcmc,
                 num_to_sample,
                 gaussian_process_list=None,
                 inner_optimizer=None,
                 discrete_pts_list=None,
                 acq_info={},
                 num_fidelity=0,
                 num_mc_iterations = 2**7,
                 points_to_sample = None):
        
        self._acq = acq
        self.__gaussian_process_mcmc = gaussian_process_mcmc
        self.__gaussian_process_list = gaussian_process_list
        self._num_fidelity = num_fidelity
        self._inner_optimizer = inner_optimizer
        self._discrete_pts_list = discrete_pts_list
        self._num_to_sample = num_to_sample
        self._num_mc_iterations = num_mc_iterations
        self._point_to_sample = points_to_sample
        self._acq_info = acq_info

        if 
        