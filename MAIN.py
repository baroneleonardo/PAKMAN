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

from examples import bayesian_optimization, auxiliary, synthetic_functions
from qaliboo import precomputed_functions, finite_domain

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
    'Levy4',  # This function implementation is probably wrong
    # Data sets
    'Query26',
    'LiGen',
    'StereoMatch',
    'LiGenTot'
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
elif objective_func_name == 'Levy4':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([1.0, 1.0, 1.0, 1.0])
    domain = finite_domain.CPPFiniteDomain.Grid(np.arange(-1, 2, 0.1),
                                                np.arange(-1, 2, 0.1),
                                                np.arange(-1, 2, 0.1),
                                                np.arange(-1, 2, 0.1))
else:
    objective_func = getattr(precomputed_functions, params.problem)
    known_minimum = objective_func.minimum
    domain = objective_func

n_initial_points = params.init
n_iterations = params.iter
n_points_per_iteration = params.points
m_domain_discretization_sample_size = params.sample_size


py_sgd_params_ps = py_optimization.GradientDescentParameters(
    max_num_steps=1000,
    max_num_restarts=3,
    num_steps_averaged=15,
    gamma=0.7,
    pre_mult=1.0,
    max_relative_change=0.02,
    tolerance=1.0e-10)

cpp_sgd_params_ps = cpp_optimization.GradientDescentParameters(
    num_multistarts=5,
    max_num_steps=6,
    max_num_restarts=3,
    num_steps_averaged=3,
    gamma=0.0,
    pre_mult=1.0,
    max_relative_change=0.1,
    tolerance=1.0e-10)

KG_gradient_descent_params = cpp_optimization.GradientDescentParameters(
    num_multistarts=20,
    max_num_steps=50,
    max_num_restarts=2,
    num_steps_averaged=4,
    gamma=0.7,
    pre_mult=1.0,
    max_relative_change=0.5,
    tolerance=1.0e-10)


# Draw initial points from domain from a Latin Hypercube(as np array)
initial_points_array = domain.sample_points_in_domain(n_initial_points)
initial_points_value = np.array([objective_func.evaluate(pt) for pt in initial_points_array])

initial_points = [data_containers.SamplePoint(pt,
                                              initial_points_value[num])
                  for num, pt in enumerate(initial_points_array)]
initial_data = data_containers.HistoricalData(dim=objective_func.dim)
initial_data.append_sample_points(initial_points)

#Queue = initial_points_array # If I want to add the queue

n_prior_hyperparameters = 1 + objective_func.dim + objective_func.n_observations
n_prior_noises = objective_func.n_observations
prior = default_priors.DefaultPrior(n_prior_hyperparameters, n_prior_noises)

# Initialization of the gaussian process
gp_loglikelihood = log_likelihood_mcmc.GaussianProcessLogLikelihoodMCMC(
    historical_data=initial_data,
    derivatives=objective_func.derivatives,  # Questo valore quando passato è 0
    prior=prior,
    chain_length=1000,
    burnin_steps=2000,
    n_hypers=N_RANDOM_WALKERS,
    noisy=True
)
gp_loglikelihood.train()


if known_minimum is not None:

    _, _, known_minimum_in_domain = domain.find_distance_index_closest_point(known_minimum)

    if not np.all(np.equal(known_minimum_in_domain, known_minimum)):
        _log.warning('Known Minimum NOT in domain')
        known_minimum = known_minimum_in_domain
_log.info(f'The minimum in the domain is:\n{known_minimum}')

###########################
# Main cycle
###########################

results = []
result_file = f'./results/prova_KG/{objective_func_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H%M")}.json'

# Algorithm 1.2: Main Stage: For `s` to `N`
for s in range(n_iterations):
    _log.info(f"{s}th iteration, "
              f"func={objective_func_name}, "
              f"q={n_points_per_iteration}")
    time1 = time.time()

    discrete_pts_list = []
    
    glb_opt_smpl = False


    cpp_gaussian_process = gp_loglikelihood.models[0]
    
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
    '''
    if glb_opt_smpl == True:
            eval_pts = np.reshape(np.append(discrete_pts_optima, eval_pts),
                                            (discrete_pts_optima.shape[0] + eval_pts.shape[0], cpp_gaussian_process.dim))   
    #discrete_pts_optima = np.reshape(eval_pts, (1, cpp_gaussian_process.dim))   
    '''
    discrete_pts_list.append(eval_pts)
    #print(f"dicrete point list = {discrete_pts_list[0].shape[0]}")    
    ps_evaluator = knowledge_gradient.PosteriorMean(gp_loglikelihood.models[0], 0)
    ps_sgd_optimizer = cpp_optimization.GradientDescentOptimizer(
        domain,
        ps_evaluator,
        cpp_sgd_params_ps
    )

    # Selection of the R restarting points
    R_points=[]
    R_points.append(np.array(domain.generate_uniform_random_points_in_domain(20*n_points_per_iteration)))
    print(R_points)
    

    # KG method
    next_points, voi = bayesian_optimization.gen_sample_from_qkg_mcmc(
        gp_loglikelihood._gaussian_process_mcmc,
        gp_loglikelihood.models,
        ps_sgd_optimizer,
        domain,
        0,
        discrete_pts_list,
        KG_gradient_descent_params,
        n_points_per_iteration,
        num_mc=2 ** 7)


    _log.info(f"Knowledge Gradient update takes {(time.time()-time1)} seconds")
    _log.info("Suggests points:")
    _log.info(next_points)

    # > ALgorithm 1.5: 5: Sample these points (z∗1 , z∗2 , · · · , z∗q)
    # > ...
    '''
    # QUEUE FOR POINTS ALREADY SAMPLED
    max_queue_size=5
    new_points = []
    for i in range(n_points_per_iteration):
        _, _, closest_point = domain.find_distance_index_closest_point(next_points[i])
        #print(f"new point === {closest_point}")
        new_points.append(closest_point)
        is_different = ~np.all(Queue == closest_point, axis=1)
        result = is_different.all()
        if result == False:
            break
    if result == False:
        _log.info("Point already sampled")
        continue
    else:
        for i in range(n_points_per_iteration):
            if Queue.shape[0] < max_queue_size:
                Queue = np.vstack((Queue, new_points[i]))
            else:
                Queue = np.vstack((Queue[1:], new_points[i]))
    
    #next_points = np.array(new_points)
'''
# Try to use new_points instead of next points in sampled points

    sampled_points = [data_containers.SamplePoint(pt,
                                                  objective_func.evaluate(pt))
                      for pt in next_points]

    time1 = time.time()
    # > ...
    # > re-train the hyperparameters of the GP by MLE
    # > and update the posterior distribution of f
    gp_loglikelihood.add_sampled_points(sampled_points)
    gp_loglikelihood.train()
    _log.info(f"Retraining the model takes {time.time() - time1} seconds")
    time1 = time.time()

    # > Algorithm 1.7: Return the argmin of the average function `μ` currently estimated in `A`

    # In the current implementation, this argmin is found
    # and returned at every interaction (called "suggested_minimum").

    _log.info("\nIteration finished successfully!")

    ####################
    # Suggested Minimum
    ####################
    # Calcola la posterior del modello retrain e trova il minimo ---> questo sarà il mio punto di minimo
    suggested_minimum = auxiliary.compute_suggested_minimum(domain, gp_loglikelihood, py_sgd_params_ps)
    # Trovo il punto più vicino al mio nel punto di minimo
    _, _, closest_point_in_domain = domain.find_distance_index_closest_point(suggested_minimum)
    computed_cost = objective_func.evaluate(closest_point_in_domain, do_not_count=True)
                 
    
    _log.info(f"The suggested minimum is:\n {suggested_minimum}")
    _log.info(f"The closest point in the finite domain is:\n {closest_point_in_domain}")
    _log.info(f"Which has a cost of:\n {computed_cost}")
    _log.info(f"Finding the suggested minimum takes {time.time() - time1} seconds")
    _log.info(f'The target function was evaluated {objective_func.evaluation_count} times')

    if known_minimum is not None:
        _log.info(f"Distance from closest point in domain to known minimum: {np.linalg.norm(closest_point_in_domain - known_minimum)}")

    error = np.linalg.norm(objective_func.min_value - computed_cost)
    error_ratio = np.abs(error/objective_func.min_value)
    _log.info(f'Error: {error}')
    _log.info(f'Error ratio: {error_ratio}')
    _log.info(f'Squared error: {np.square(error)}')
    
    results.append(
        dict(
            iteration=s,
            n_initial_points=n_initial_points,
            q=n_points_per_iteration,
            m=m_domain_discretization_sample_size,
            target=objective_func_name,
            suggested_minimum=suggested_minimum.tolist(),
            known_minimum=known_minimum.tolist(),
            closest_point_in_domain=closest_point_in_domain.tolist(),
            computed_cost=float(computed_cost),
            n_evaluations=objective_func.evaluation_count,
            error=error,
            error_ratio=error_ratio
        )
    )

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    '''
    if error < 0.0000001:
        _log.info(f'Error is small enough. Exiting cycle at iteration {s}')
        break
    '''
_log.info("\nOptimization finished successfully!")