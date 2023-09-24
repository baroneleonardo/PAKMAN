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

from examples import bayesian_optimization, auxiliary, synthetic_functions
from qaliboo import precomputed_functions, finite_domain

logging.basicConfig(level=logging.NOTSET)
_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)

###########################
# Constants
###########################
N_RANDOM_WALKERS = 2 ** 4
AVAILABLE_PROBLEMS = [
    # Toy problems:
    'ParabolicMinAtOrigin',
    'ParabolicMinAtTwoAndThree',
    # Benchmark functions:
    'Hartmann3',
    # 'Levy4',  # This function implementation is probably wrong
    # Data sets
    'Query26',
    'LiGen',
    'StereoMatch'
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
parser.add_argument('--iter', '-n', help='Number of iterations', type=int, default=10)
parser.add_argument('--points', '-q', help='Points per iteration (the `q` parameter)', type=int, default=7)
parser.add_argument('--sample_size', '-m', help='GP sample size (`M` parameter)', type=int, default=12)

params = parser.parse_args()

#############################################
# Collection of target functions and domains
#############################################
# objective_func_name = 'Parabolic with minimum at (2, 3)'
# objective_func = synthetic_functions.ParabolicMinAtTwoAndThree()
# known_minimum = np.array([2.0, 3.0])
# domain = finite_domain.FiniteDomain.Grid(np.arange(-5, 5, 0.1),
#                                          np.arange(-5, 5, 0.1))

# objective_func_name = 'Hartmann3'
# objective_func = synthetic_functions.Hartmann3()
# known_minimum = None
# domain = finite_domain.FiniteDomain.Grid(np.arange(0, 1, 0.01),
#                                          np.arange(0, 1, 0.01),
#                                          np.arange(0, 1, 0.01))

# objective_func_name = 'Query26'
# objective_func = precomputed_functions.Query26
# known_minimum = objective_func.minimum
# domain = objective_func
# # SUGGESTED FOR Query26:
# N_INITIAL_POINTS = 4
# N_ITERATIONS = 5
# N_POINTS_PER_ITERATION = 4  # The q- parameter
# M_DOMAIN_DISCRETIZATION_SAMPLE_SIZE = 8  # M parameter
# N_RANDOM_WALKERS = 2 ** 4

# # SUGGESTED FOR LiGen:
# N_INITIAL_POINTS = 7
# N_ITERATIONS = 10
# N_POINTS_PER_ITERATION = 7  # The q- parameter
# M_DOMAIN_DISCRETIZATION_SAMPLE_SIZE = 14  # M parameter
# N_RANDOM_WALKERS = 2 ** 4  # Originally fixed at 2 ** 4 = 16

# # SUGGESTED FOR StereoMatch:
# N_INITIAL_POINTS = 5
# N_ITERATIONS = 15
# N_POINTS_PER_ITERATION = 5  # The q- parameter
# M_DOMAIN_DISCRETIZATION_SAMPLE_SIZE = 10  # M parameter
# N_RANDOM_WALKERS = 2 ** 4  # Originally fixed at 2 ** 4 = 16


#########################################
# Initializing variables from parameters
#########################################
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
# elif objective_func_name == 'Levy4':  # Levy4 implementation is probably broken
#     objective_func = getattr(synthetic_functions, params.problem)()
#     known_minimum = np.array([1, 1, 1, 1])
#     domain = finite_domain.CPPFiniteDomain.Grid(np.arange(-5, 5, 0.1),
#                                                 np.arange(-5, 5, 0.1),
#                                                 np.arange(-5, 5, 0.1),
#                                                 np.arange(-5, 5, 0.1))
else:
    objective_func = getattr(precomputed_functions, params.problem)
    known_minimum = objective_func.minimum
    domain = objective_func

n_initial_points = params.init
n_iterations = params.iter
n_points_per_iteration = params.points
m_domain_discretization_sample_size = params.sample_size


###############################
# Initializing utility objects
###############################

# Various Gradient Descent parameters
py_sgd_params_ps = py_optimization.GradientDescentParameters(
    max_num_steps=1000,
    max_num_restarts=3,
    num_steps_averaged=15,
    gamma=0.7,
    pre_mult=1.0,
    max_relative_change=0.02,
    tolerance=1.0e-10)

cpp_sgd_params_ps = cpp_optimization.GradientDescentParameters(
    num_multistarts=1,
    max_num_steps=6,
    max_num_restarts=1,
    num_steps_averaged=3,
    gamma=0.0,
    pre_mult=1.0,
    max_relative_change=0.1,
    tolerance=1.0e-10)

KG_gradient_descent_params = cpp_optimization.GradientDescentParameters(
    num_multistarts=200,
    max_num_steps=50,
    max_num_restarts=2,
    num_steps_averaged=4,
    gamma=0.7,
    pre_mult=1.0,
    max_relative_change=0.5,
    tolerance=1.0e-10)

###########################
# Starting up the MCMC...
###########################

# > Algorithm 1.1: Initial Stage: draw `I` initial samples from a latin hypercube design in `A` (domain)

# Draw initial points from domain (as np array)
initial_points_array = domain.sample_points_in_domain(n_initial_points)
# Evaluate function in initial points
initial_points_value = np.array([objective_func.evaluate(pt) for pt in initial_points_array])
# Build points using custom container class
initial_points = [data_containers.SamplePoint(pt,
                                              initial_points_value[num])
                  for num, pt in enumerate(initial_points_array)]

# Build historical data container
initial_data = data_containers.HistoricalData(dim=objective_func.dim)

initial_data.append_sample_points(initial_points)

# initialize the model
# (It is completely unclear what these parameters mean)
n_prior_hyperparameters = 1 + objective_func.dim + objective_func.n_observations
n_prior_noises = objective_func.n_observations
prior = default_priors.DefaultPrior(n_prior_hyperparameters, n_prior_noises)

# noisy = False means the underlying function being optimized is noise-free
gp_loglikelihood = log_likelihood_mcmc.GaussianProcessLogLikelihoodMCMC(
    historical_data=initial_data,
    derivatives=objective_func.derivatives,
    prior=prior,
    chain_length=1000,
    burnin_steps=2000,
    n_hypers=N_RANDOM_WALKERS,
    noisy=True
)
gp_loglikelihood.train()

# If available, find point in domain closest to the minimum
if known_minimum is not None:

    _, _, known_minimum_in_domain = domain.find_distance_index_closest_point(known_minimum)

    if not np.all(np.equal(known_minimum_in_domain, known_minimum)):
        _log.warning('Known Minimum NOT in domain')
        known_minimum = known_minimum_in_domain

###########################
# Main cycle
###########################

results = []
result_file = f'./results/simplified_runs/{objective_func_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H%M")}.json'

# Algorithm 1.2: Main Stage: For `s` to `N`
for s in range(n_iterations):
    _log.info(f"{s}th iteration, "
              f"func={objective_func_name}, "
              f"q={n_points_per_iteration}")
    time1 = time.time()

    # > The q-KG algorithm will reduce to the parallel EI algorithm
    # > if function evaluations are noise-free
    # > and the final recommendation is restricted to the previous sampling decisions.

    # Run KG
    # > Algorithm 1.4: Solve the main equation:
    # > (find points (z∗1 , z∗2 , · · · , z∗q) that maximize the q-KG factor

    # > Par 5.3:
    # > In this paper, the discrete set A_n is not chosen statically,
    # > but evolves over time: specifically, we suggest drawing M samples
    # > from the global optima of the posterior distribution of the Gaussian
    # > process. This sample set, denoted by A_n^M, is then extended
    # > by the locations of previously sampled [=evaluated] points x(1:n)
    # > and the set of candidate points z(1:q)

    # So, the infinite domain A is replaced by a discrete set of points A_n (see above)
    discrete_pts_list = []

    qEI_next_points, _ = bayesian_optimization.qEI_generate_next_points_using_mcmc(
        gaussian_process_mcmc=gp_loglikelihood._gaussian_process_mcmc,
        search_domain=domain,
        gd_params=KG_gradient_descent_params,
        q=m_domain_discretization_sample_size,
        mc_iterations=2 ** 10)

    for i, cpp_gaussian_process in enumerate(gp_loglikelihood.models):  # What are these models?

        eval_pts = domain.generate_uniform_random_points_in_domain(int(1e2))
        eval_pts = np.reshape(np.append(eval_pts,
                                        (cpp_gaussian_process.get_historical_data_copy()).points_sampled[:, :(gp_loglikelihood.dim)]),
                              (eval_pts.shape[0] + cpp_gaussian_process.num_sampled, cpp_gaussian_process.dim))

        test = np.zeros(eval_pts.shape[0])
        ps_evaluator = knowledge_gradient.PosteriorMean(cpp_gaussian_process)
        for i, pt in enumerate(eval_pts):
            ps_evaluator.set_current_point(pt.reshape((1, gp_loglikelihood.dim)))
            test[i] = -ps_evaluator.compute_objective_function()

        initial_point = eval_pts[np.argmin(test)]

        ps_sgd_optimizer = cpp_optimization.GradientDescentOptimizer(
            domain,
            ps_evaluator,
            cpp_sgd_params_ps
        )

        report_point = knowledge_gradient.posterior_mean_optimization(
            ps_sgd_optimizer,
            initial_guess=initial_point,
            max_num_threads=4
        )

        discrete_pts_optima = np.reshape(np.append(qEI_next_points, report_point),
                                         (qEI_next_points.shape[0] + 1, cpp_gaussian_process.dim))
        discrete_pts_list.append(discrete_pts_optima)

    ps_evaluator = knowledge_gradient.PosteriorMean(gp_loglikelihood.models[0], 0)
    ps_sgd_optimizer = cpp_optimization.GradientDescentOptimizer(
        domain,
        ps_evaluator,
        cpp_sgd_params_ps
    )
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
    suggested_minimum = auxiliary.compute_suggested_minimum(domain, gp_loglikelihood, py_sgd_params_ps)
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

    if error < 0.0000001:
        _log.info(f'Error is small enough. Exiting cycle at iteration {s}')
        break

_log.info("\nOptimization finished successfully!")
