from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
from sys import argv
import time


from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import PosteriorMeanMCMC
from moe.optimal_learning.python.cpp_wrappers.log_likelihood_mcmc import GaussianProcessLogLikelihoodMCMC as cppGaussianProcessLogLikelihoodMCMC
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentParameters as cppGradientDescentParameters
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentOptimizer as cppGradientDescentOptimizer
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient import posterior_mean_optimization, PosteriorMean

from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.repeated_domain import RepeatedDomain
from moe.optimal_learning.python.default_priors import DefaultPrior

from moe.optimal_learning.python.python_version.optimization import GradientDescentParameters as pyGradientDescentParameters
from moe.optimal_learning.python.python_version.optimization import GradientDescentOptimizer as pyGradientDescentOptimizer
from moe.optimal_learning.python.python_version.optimization import multistart_optimize as multistart_optimize

from examples import bayesian_optimization
from examples import synthetic_functions

# arguments for calling this script:
# python main.py [obj_func_name] [method_name] [num_to_sample] [job_id] [hesbo_flag] [effective_dim]
# example: python main.py Branin KG 4 1
# example: python main.py Branin KG 4 1 HeSBO 5
# you can define your own obj_function and then just change the objective_func object below, and run this script.

######################
#
# Parsing Arguments
#
######################
# NOTE: argv[0] is just the name of the script
obj_func_name = argv[1]  # Exact name of the objective function
# TODO: This "objective_function" object is actually a PROBLEM DEFINITION
objective_func = getattr(synthetic_functions, obj_func_name, None)  # Getting class from module
if objective_func is None:
    raise ValueError(f'The objective function {obj_func_name} can not be found!')
else:  # Instantiating objective function
    objective_func = objective_func()

method = argv[2]  # 'KG' or 'EI'
if method.upper() not in ('KG', 'EI'):
    raise ValueError(f'Unknown method "{method}"!')

num_to_sample = int(argv[3])  # The "q" parameter

job_id = int(argv[4])  # TODO: This parameter seems fairly useless

# FIXED ARGUMENT (TODO: This should be a script argument)
NUM_FUNC_EVAL = 16  # Total number of function evaluations

######################
#
# Initialization
#
######################
# We will at most evaluate the function NUM_FUNC_EVAL times
num_iteration = NUM_FUNC_EVAL // num_to_sample + 1

search_domain = objective_func.get_search_domain()

# Get the initial data
# NOTE qKG: 1: Initial Stage: draw I initial samples from a latin hypercube design in A, x(i) for i = 1, . . . , I
init_pts = search_domain.generate_uniform_random_points_in_domain(objective_func._num_init_pts)

# Evaluate function in initial points
init_pts_value = np.array([objective_func.evaluate(pt) for pt in init_pts])

init_data = HistoricalData(dim=objective_func._dim, num_derivatives=objective_func.n_derivatives)
init_data.append_sample_points([SamplePoint(pt,
                                            [init_pts_value[num, i] for i in objective_func.observations])
                                for num, pt in enumerate(init_pts)])

# initialize the model
# (It is completely unclear what these parameters mean)
n_prior_hyperparameters = 1 + objective_func._dim + objective_func.n_observations
n_prior_noises = objective_func.n_observations
prior = DefaultPrior(n_prior_hyperparameters, n_prior_noises)

# noisy = False means the underlying function being optimized is noise-free
gp_loglikelihood = cppGaussianProcessLogLikelihoodMCMC(historical_data=init_data,
                                                       derivatives=objective_func.derivatives,
                                                       prior=prior,
                                                       chain_length=1000,
                                                       burnin_steps=2000,
                                                       n_hypers=2 ** 4,
                                                       noisy=False)
gp_loglikelihood.train()

# Various Gradient Descent parameters
py_sgd_params_ps = pyGradientDescentParameters(max_num_steps=1000,
                                               max_num_restarts=3,
                                               num_steps_averaged=15,
                                               gamma=0.7,
                                               pre_mult=1.0,
                                               max_relative_change=0.02,
                                               tolerance=1.0e-10)

cpp_sgd_params_ps = cppGradientDescentParameters(num_multistarts=1,
                                                 max_num_steps=6,
                                                 max_num_restarts=1,
                                                 num_steps_averaged=3,
                                                 gamma=0.0,
                                                 pre_mult=1.0,
                                                 max_relative_change=0.1,
                                                 tolerance=1.0e-10)

KG_gradient_descent_params = cppGradientDescentParameters(num_multistarts=200,
                                                          max_num_steps=50,
                                                          max_num_restarts=2,
                                                          num_steps_averaged=4,
                                                          gamma=0.7,
                                                          pre_mult=1.0,
                                                          max_relative_change=0.5,
                                                          tolerance=1.0e-10)

for n in range(num_iteration):
    print(f"{method}, {job_id}th job, {n}th iteration, func={obj_func_name}, q={num_to_sample}")
    time1 = time.time()

    if method == 'KG':
        discrete_pts_list = []

        qEI_next_points, _ = bayesian_optimization.qEI_generate_next_points_using_mcmc(gp_loglikelihood._gaussian_process_mcmc,
                                                                                       search_domain,
                                                                                       KG_gradient_descent_params,
                                                                                       10,  # TODO: Why 10?
                                                                                       mc_iterations=2 ** 10)

        for i, cpp_gp in enumerate(gp_loglikelihood.models):  # What are these models?

            eval_pts = search_domain.generate_uniform_random_points_in_domain(int(1e3))
            eval_pts = np.reshape(np.append(eval_pts,
                                            (cpp_gp.get_historical_data_copy()).points_sampled[:, :(gp_loglikelihood.dim - objective_func._num_fidelity)]),
                                  (eval_pts.shape[0] + cpp_gp.num_sampled, cpp_gp.dim-objective_func._num_fidelity))

            test = np.zeros(eval_pts.shape[0])
            ps_evaluator = PosteriorMean(cpp_gp, objective_func._num_fidelity)
            for i, pt in enumerate(eval_pts):
                ps_evaluator.set_current_point(pt.reshape((1, gp_loglikelihood.dim - objective_func._num_fidelity)))
                test[i] = -ps_evaluator.compute_objective_function()

            initial_point = eval_pts[np.argmin(test)]

            ps_sgd_optimizer = cppGradientDescentOptimizer(search_domain, ps_evaluator, cpp_sgd_params_ps)
            report_point = posterior_mean_optimization(ps_sgd_optimizer, initial_guess = initial_point, max_num_threads = 4)


            discrete_pts_optima = np.reshape(np.append(qEI_next_points, report_point),
                                             (qEI_next_points.shape[0] + 1, cpp_gp.dim-objective_func._num_fidelity))
            discrete_pts_list.append(discrete_pts_optima)

        ps_evaluator = PosteriorMean(gp_loglikelihood.models[0], objective_func._num_fidelity)
        ps_sgd_optimizer = cppGradientDescentOptimizer(search_domain, ps_evaluator, cpp_sgd_params_ps)
        # KG method
        next_points, voi = bayesian_optimization.gen_sample_from_qkg_mcmc(gp_loglikelihood._gaussian_process_mcmc,
                                                                          gp_loglikelihood.models,
                                                                          ps_sgd_optimizer,
                                                                          search_domain,
                                                                          objective_func._num_fidelity,
                                                                          discrete_pts_list,
                                                                          KG_gradient_descent_params,
                                                                          num_to_sample,
                                                                          num_mc=2 ** 7)

    else:  # method == 'EI':
        next_points, voi = bayesian_optimization.gen_sample_from_qei(gp_loglikelihood.models[0],
                                                                     search_domain,
                                                                     KG_gradient_descent_params,
                                                                     num_to_sample,
                                                                     num_mc=2 ** 10)

    print(f"{method} takes {(time.time()-time1)} seconds")
    print("Suggests points:")
    print(next_points)

    sampled_points = [SamplePoint(pt,
                                  objective_func.evaluate(pt)[objective_func.observations])
                      for pt in next_points]

    # retrain the model
    time1 = time.time()

    gp_loglikelihood.add_sampled_points(sampled_points)
    gp_loglikelihood.train()

    print(f"Retraining the model takes {time.time() - time1} seconds")
    time1 = time.time()

    # report the point
    if method == 'KG':
        eval_pts = search_domain.generate_uniform_random_points_in_domain(int(1e4))
        eval_pts = np.reshape(np.append(eval_pts, (gp_loglikelihood.get_historical_data_copy()).points_sampled[:, :(gp_loglikelihood.dim - objective_func._num_fidelity)]),
                              (eval_pts.shape[0] + gp_loglikelihood._num_sampled, gp_loglikelihood.dim - objective_func._num_fidelity))

        post_mean = PosteriorMeanMCMC(gp_loglikelihood.models, objective_func._num_fidelity)
        test = np.zeros(eval_pts.shape[0])
        for i, pt in enumerate(eval_pts):
            post_mean.set_current_point(pt.reshape((1, gp_loglikelihood.dim - objective_func._num_fidelity)))
            test[i] = -post_mean.compute_objective_function()
        initial_point = eval_pts[np.argmin(test)].reshape((1, gp_loglikelihood.dim - objective_func._num_fidelity))

        py_repeated_search_domain = RepeatedDomain(num_repeats=1, domain=search_domain)
        ps_mean_opt = pyGradientDescentOptimizer(py_repeated_search_domain, post_mean, py_sgd_params_ps)
        report_point = multistart_optimize(ps_mean_opt, initial_point, num_multistarts = 1)[0]

    else:  # method == 'EI':
        cpp_gp = gp_loglikelihood.models[0]
        report_point = (cpp_gp.get_historical_data_copy()).points_sampled[np.argmin(cpp_gp._points_sampled_value[:, 0])]

    report_point = report_point.ravel()
    report_point = np.concatenate((report_point, np.ones(objective_func._num_fidelity)))

    print("\nOptimization finished successfully!")
    print(f"The recommended point: {report_point}")
    print(f"The recommended integer point: {np.round(report_point).astype(int)}")
    print(f"Finding the recommended point takes {time.time()-time1} seconds")
    print(f" {method}, VOI {voi}, best so far {objective_func.evaluate_true(report_point)[0]}")
