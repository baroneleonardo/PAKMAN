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


class UtilityFunction(object):
    def __init__(self, 
                 acq, 
                 gaussian_process_mcmc,
                 num_to_sample,
                 domain=None,
                 gaussian_process_list=None,
                 discrete_pts_list=None,
                 num_fidelity=0,
                 num_mc_iterations = 2**7):
        
        self._num_to_sample = num_to_sample

        if acq=='EI' or acq =='EI_ml':
            self._util = EI.ExpectedImprovementMCMC(gaussian_process_mcmc=gaussian_process_mcmc,
                                                    num_to_sample=num_to_sample,
                                                    num_mc_iterations=num_mc_iterations)
        
        elif acq == 'KG' or acq == 'KG_ml':
            
            ps_evaluator = knowledge_gradient.PosteriorMean(gaussian_process_list[0], 0)
            cpp_sgd_params_ps = cpp_optimization.GradientDescentParameters(
            num_multistarts=5,
            max_num_steps=6,
            max_num_restarts=3,
            num_steps_averaged=3,
            gamma=0.0,
            pre_mult=1.0,
            max_relative_change=0.2,
            tolerance=1.0e-10)
            ps_sgd_optimizer = cpp_optimization.GradientDescentOptimizer(
            domain,
            ps_evaluator,
            cpp_sgd_params_ps
            )

            self._util = KG.KnowledgeGradientMCMC(gaussian_process_mcmc=gaussian_process_mcmc,
                                            gaussian_process_list=gaussian_process_list,
                                            inner_optimizer=ps_sgd_optimizer,
                                            discrete_pts_list=discrete_pts_list,
                                            num_to_sample=num_to_sample,
                                            num_mc_iterations=num_mc_iterations,
                                            num_fidelity=num_fidelity)
            
        else:
            raise KeyError('Select a valid model!')
        
    @property
    def util(self):
        return self._util
    @property
    def num_to_sample(self):
        return self._num_to_sample