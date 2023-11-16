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
from qaliboo.utils import UtilityFunction 
from concurrent.futures import ProcessPoolExecutor

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
    
    def space_of_evaluation(self, 
                            domain, 
                            gp_loglikelihood,
                            m_domain_discretization_sample_size,
                            objective_func,
                            glb_opt_smpl):
        
        cpp_gaussian_process = gp_loglikelihood.models[0]
        discrete_pts_list = []
        
        if glb_opt_smpl == True:
                init_points = domain.generate_uniform_random_points_in_domain(int(1e2))
                discrete_pts_optima = random_features.sample_from_global_optima(cpp_gaussian_process, 
                                                                                100, 
                                                                                objective_func.search_domain, 
                                                                                init_points, 
                                                                                m_domain_discretization_sample_size
                                                                                )
                eval_pts = np.reshape(np.append(discrete_pts_optima,
                                                (cpp_gaussian_process.get_historical_data_copy()).points_sampled[:, :(gp_loglikelihood.dim)]),
                                                (discrete_pts_optima.shape[0] + cpp_gaussian_process.num_sampled, cpp_gaussian_process.dim))
        else:
            eval_pts = domain.generate_uniform_random_points_in_domain(int(m_domain_discretization_sample_size))  # Sample continuous
            #eval_pts = domain.sample_points_in_domain(sample_size=int(m_domain_discretization_sample_size), allow_previously_sampled=True) # Sample discrete
        
            eval_pts = np.reshape(np.append(eval_pts,
                                            (cpp_gaussian_process.get_historical_data_copy()).points_sampled[:, :(gp_loglikelihood.dim)]),
                                (eval_pts.shape[0] + cpp_gaussian_process.num_sampled, cpp_gaussian_process.dim))

        discrete_pts_list.append(eval_pts)

        return discrete_pts_list


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
            #TODO define this section
            ml_model = ML_model(X_data=initial_points_array,
                                y_data=np.array([self._objective_func.evaluate_time(pt) for pt in initial_points_array]), 
                                X_ub=2.5) # Set this value if you are intrested in I(T(X) < X_ub)

        n_prior_hyperparameters = 1 + self._objective_func.dim + self._objective_func.n_observations
        n_prior_noises = self._objective_func.n_observations
        prior = default_priors.DefaultPrior(n_prior_hyperparameters, n_prior_noises)

        # Initialization of the gaussian process
        gp_loglikelihood = log_likelihood_mcmc.GaussianProcessLogLikelihoodMCMC(
            historical_data=initial_data,
            derivatives=self._objective_func.derivatives,  # Questo valore quando passato è 0
            prior=prior,
            chain_length=1000,
            burnin_steps=2000,
            n_hypers=1,
            noisy=False
        )

        gp_loglikelihood.train()
        
        for s in range(n_iter):
            _log.info(f"{s}th iteration, "
              f"func={self._objective_func.name}, "
              f"q={q}")
            
            time1 = time.time()
            
            if acq=='KG' or acq=='KG_ml':

                discrete_pts_list = self.space_of_evaluation(self._domain,
                                                        gp_loglikelihood,
                                                        m_dom_disc,
                                                        self._objective_func,
                                                        True)
            
                util_func = UtilityFunction(acq,
                                            gp_loglikelihood._gaussian_process_mcmc,
                                            q,
                                            self._domain,
                                            gp_loglikelihood.models,
                                            discrete_pts_list).util
                
            elif acq == 'EI' or acq=='EI_ml':
                util_func = UtilityFunction(acq,
                                            gp_loglikelihood._gaussian_process_mcmc,
                                            q)

            num_restarts = 20    
            with ProcessPoolExecutor(max_workers=5) as executor:
                res = list(executor.map(optimize_point, range(num_restarts)))
        

            report_point, kg_list = zip(*res)

            index = np.argmax(kg_list)
            next_points = report_point[index]


    def optimize_point(self, util_func, seed, n_iter_sa = 20):

        init_point = np.array(self._domain.generate_uniform_random_points_in_domain(n_points_per_iteration))
        
        new_point = SA.simulated_annealing(self._domain, util_func, init_point, n_iter_sa, initial_temperature, max_relative_change)
        
        new_point = sga.sga_kg(kg, domain, new_point)
        
        kg.set_current_point(new_point)

        #identity = ml_model.nascent_minima(new_point)
        if use_ml==True:    
            identity = ml_model.identity(new_point)
        else:
            identity=1
        kg_value = kg.compute_knowledge_gradient_mcmc()*identity 
        
        return new_point, kg_value
    







