import datetime
import logging
import time
import json
import argparse
import os
import numpy as np
import numpy.linalg
import random

from moe.optimal_learning.python import data_containers
from moe.optimal_learning.python.cpp_wrappers import log_likelihood_mcmc, optimization as cpp_optimization, knowledge_gradient
from moe.optimal_learning.python.python_version import optimization as py_optimization
from moe.optimal_learning.python import default_priors
from moe.optimal_learning.python import random_features
from moe.optimal_learning.python.cpp_wrappers import knowledge_gradient_mcmc as KG
from examples import  auxiliary
from qaliboo import sga_kg as sga
from qaliboo.machine_learning_models import ML_model
from concurrent.futures import ProcessPoolExecutor
from qaliboo import aux
logging.basicConfig(level=logging.NOTSET)
_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)

class ParallelMaliboo:
    def __init__(self, n_initial_points: int = 10, n_iterations: int = 30, batch_size:int = 4,
                 m_domain_discretization: int= 30, objective_func = None, lb: float=None, 
                 ub: float=None, nm:bool=False, optimum_sample:bool=False, n_restarts:int = 15):

        self._n_initial_points = n_initial_points
        self._n_iterations = n_iterations
        self._q = batch_size
        self._m = m_domain_discretization
        self._domain = objective_func
        self._objective_func = objective_func
        self._ub = ub
        self._lb = lb
        self._nm = nm
        self._optimum_sample = optimum_sample
        self._n_restarts=n_restarts
        
        self._py_sgd_params_ps = py_optimization.GradientDescentParameters(
            max_num_steps=1000, max_num_restarts=3,
            num_steps_averaged=15, gamma=0.7, pre_mult=1.0,
            max_relative_change=0.02, tolerance=1.0e-10)

        self._cpp_sgd_params_ps = cpp_optimization.GradientDescentParameters(
            num_multistarts=5, max_num_steps=6, max_num_restarts=3,
            num_steps_averaged=3, gamma=0.0, pre_mult=1.0,
            max_relative_change=0.2, tolerance=1.0e-10)

        self._global_time=0

        initial_points_array = self._domain.sample_points_in_domain(n_initial_points)
        
        print(len(initial_points_array))
        
        initial_points_value, initial_points_index, initial_points_time = self.evaluate_next_points(initial_points_array)
        print(initial_points_value)
        '''
        initial_points_array = self._domain.sample_points_in_domain(n_initial_points)
        initial_points_value = np.zeros(n_initial_points)
        initial_points_index = np.zeros(n_initial_points)
        initial_points_time = np.zeros(n_initial_points)
        count=0
        
        for pt in initial_points_array:
            poi_v, poi_i, poi_t = objective_func.evaluate(pt) 
            initial_points_value[count]= poi_v
            initial_points_index[count]=poi_i
            initial_points_time[count] = np.array(poi_t)
            count= count+1
        '''
        initial_points = [data_containers.SamplePoint(pt,
                                              initial_points_value[num])
                  for num, pt in enumerate(initial_points_array)]
        
        initial_data = data_containers.HistoricalData(dim=objective_func.dim)
        initial_data.append_sample_points(initial_points)

        self._min_evaluated = np.min(initial_points_value)

        self._use_ml=False
        if (ub is not None) or (lb is not None) or (nm is not False):
            self._use_ml = True
            _log.info("You have selected an acquisition function with ML integrated")
            self._ml_model = ML_model(X_data=initial_points_array, 
                        y_data=np.array([objective_func.evaluate_time(pt) for pt in initial_points_array]), 
                        X_ub=ub,
                        X_lb=lb) 
        else: _log.info("Without ML model")

        self._gp_loglikelihood = log_likelihood_mcmc.GaussianProcessLogLikelihoodMCMC(
            historical_data=initial_data,
            derivatives=objective_func.derivatives,  # Questo valore quando passato è 0
            prior=default_priors.DefaultPrior(1 + objective_func.dim + objective_func.n_observations, objective_func.n_observations),
            chain_length=1000,
            burnin_steps=2000,
            n_hypers=1,
            noisy=False
        )
        self._gp_loglikelihood.train()

    # Syncronous Knoledge Gradient optimization
    def sync_optimization(self):
        for s in range(self._n_iterations):
            self.iteration_step(s)
        _log.info("\nOptimization finished successfully")

    # Optimization Iteration
    def iteration_step(self, iteration_number):
        _log.info(f"{iteration_number}th iteration,  "f"q={self._q}")
        init_alg_time = time.time()
        time1 = time.time()
        cpp_gaussian_process = self._gp_loglikelihood.models[0]
        
        # Sampling of the domain discretization (by or not sampling from the global optima)
        discrete_pts_list = self.global_optimum_sample(self._optimum_sample,cpp_gaussian_process) 
        # Definition of the acquisition function (Knoledge Gradient)
        ps_evaluator = knowledge_gradient.PosteriorMean(cpp_gaussian_process, 0)
        ps_sgd_optimizer = cpp_optimization.GradientDescentOptimizer(self._domain,ps_evaluator,self._cpp_sgd_params_ps)        
        kg = KG.KnowledgeGradientMCMC(gaussian_process_mcmc=self._gp_loglikelihood._gaussian_process_mcmc,
                                        gaussian_process_list=self._gp_loglikelihood.models,num_fidelity=0,inner_optimizer=ps_sgd_optimizer,
                                        discrete_pts_list=discrete_pts_list,num_to_sample=self._q,num_mc_iterations=2**7,points_to_sample=None)
        
        # Multistart Knoledge Gradient Optimization
        seeds = np.random.randint(0, 10000, size=self._n_restarts)
        report_point=[]
        kg_list = []
        for i in range(self._n_restarts):
            new_point, kg_value = self.optimize_point(seeds[i], kg)
            report_point.append(new_point)
            kg_list.append(kg_value)
        index = np.argmax(kg_list)
        next_points = report_point[index]
        _log.info(f"Knowledge Gradient update takes {(time.time()-time1)} seconds")
        
        # Evaluation objective function 
        next_points_value, next_points_index, next_points_time = self.evaluate_next_points(next_points)
        max_time = max(next_points_time)
        self._min_evaluated = min([self._min_evaluated, np.min(next_points_value)])
        sampled_points = [data_containers.SamplePoint(pt, next_points_value[num])
                            for num, pt in enumerate(next_points)]

        # Update the ML model
        if self._use_ml:
            target = np.array([self._objective_func.evaluate_time(pt) for pt in next_points])
            self._ml_model.update(next_points, target)

        # Update the Gaussian Process 
        self.update_gp_loglikelihood(sampled_points)

        suggested_minimum = self.find_suggested_minimum()
        alg_time = time.time() - init_alg_time
        self._global_time += max_time + alg_time

        self.log_iteration_result(suggested_minimum, target)

    # Evaluate the objective function
    def evaluate_next_points(self, next_points):
        next_points_value = np.zeros(self._q)
        next_points_index = np.zeros(self._q)
        next_points_time = np.zeros(self._q)
        for count, pt in enumerate(next_points):
            poi_v, poi_i, poi_t = self._objective_func.evaluate(pt) 
            next_points_value[count] = poi_v
            next_points_index[count] = poi_i
            next_points_time[count] = np.array(poi_t)
        return next_points_value, next_points_index, next_points_time

    # Update the Gaussian Process with the new sampled points
    def update_gp_loglikelihood(self, sampled_points):
        time1 = time.time()
        self._gp_loglikelihood.add_sampled_points(sampled_points)
        self._gp_loglikelihood.train()
        _log.info(f"Retraining the model takes {time.time() - time1} seconds")

    # Compute the minimum of the posterior distribution
    def find_suggested_minimum(self):
        time1 = time.time()
        suggested_minimum = auxiliary.compute_suggested_minimum(self._domain, self._gp_loglikelihood, self._py_sgd_params_ps)
        _, _, closest_point_in_domain = self._domain.find_distance_index_closest_point(suggested_minimum)
        computed_cost = self._objective_func.evaluate(closest_point_in_domain, do_not_count=True)[0]
        _log.info(f"Finding the suggested minimum takes {time.time() - time1} seconds")
        return computed_cost

    # Log of the results of the run
    def log_iteration_result(self, computed_cost, target):
        _log.info("\nIteration finished successfully!")
        _log.info(f"Which has a cost of:\n {computed_cost}")
        _log.info(f"Cost of the minimum evaluated:\n {self._min_evaluated}")
        _log.info(f'The target function was evaluated {self._objective_func.evaluation_count} times')
        _log.info(f'Unfeasable Point:{self._ml_model.out_count(target)}')
        error = np.linalg.norm(self._objective_func.min_value - computed_cost)
        error_ratio = np.abs(error / self._objective_func.min_value)
        _log.info(f'Error: {error}')
        _log.info(f'Error ratio: {error_ratio}')
        _log.info(f'Optimizer Time: {self._global_time}')

    # Discretization of the domain
    def global_optimum_sample(self, b, cpp_gaussian_process):
        discrete_pts_list = []
        if b==True:
            init_points = self._domain.generate_uniform_random_points_in_domain(int(1e2))
            discrete_pts_optima = random_features.sample_from_global_optima(cpp_gaussian_process,100, 
                                                                            self._objective_func.search_domain,init_points,self._m)
            eval_pts = np.reshape(np.append(discrete_pts_optima,
                                            (cpp_gaussian_process.get_historical_data_copy()).points_sampled[:, :(self._gp_loglikelihood.dim)]),
                                            (discrete_pts_optima.shape[0] + cpp_gaussian_process.num_sampled, cpp_gaussian_process.dim))
        else:
            eval_pts = self._domain.generate_uniform_random_points_in_domain(int(self._m))  # Sample continuous
            eval_pts = np.reshape(np.append(eval_pts,
                                            (cpp_gaussian_process.get_historical_data_copy()).points_sampled[:, :(self._gp_loglikelihood.dim)]),
                                (eval_pts.shape[0] + cpp_gaussian_process.num_sampled, cpp_gaussian_process.dim))
        discrete_pts_list.append(eval_pts) 
        return discrete_pts_list
    
    # Optimization of the aquisitionfunction
    def optimize_point(self, seed, kg):
        np.random.seed(seed)
        init_point = np.array(self._domain.generate_uniform_random_points_in_domain(self._q))
        new_point=init_point
        # Stocastic Gradient Ascent
        new_point = sga.sga_kg_ml(kg, self._domain, new_point, self._ml_model)
        kg.set_current_point(new_point)
        # Machine Learning Penalization method
        identity = 1
        if self._nm==True:    
            identity *= self._ml_model.nascent_minima(new_point)
        if (self._ub is not None) or (self._lb is not None):
            identity*=self._ml_model.exponential_penality(new_point, 4)
        kg_value = kg.compute_knowledge_gradient_mcmc()*identity 
        return new_point, kg_value  
  

