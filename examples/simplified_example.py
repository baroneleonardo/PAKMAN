import time

import sklearn.datasets
import numpy as np

from moe.optimal_learning.python import data_containers
from moe.optimal_learning.python.cpp_wrappers import log_likelihood_mcmc
from moe.optimal_learning.python import default_priors

from moe.build import GPP as moe_cpp

from examples import synthetic_functions, bayesian_optimization

###########################
# Constants
###########################
N_INITIAL_POINTS = 5
N_ITERATIONS = 5
N_POINTS_PER_ITERATION = 3  # The q- parameter
MODE = 'EI'  # 'EI' vs 'KG'

TARGET_COLUMN = 'target'

###########################
# Target function
###########################
objective_func_name = 'Parabolic with minimum at (2, 3)'
objective_func = synthetic_functions.ParabolicMinAtTwoAndThree()

###############################
# Initializing utility objects
###############################

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

###########################
# Loading data in domain
###########################

precomputed_sample_df = sklearn.datasets.load_diabetes(as_frame=True)['frame']

# No need for bounds, BUT you might need to specify column names for first derivative, second derivative, etc.
domain = moe_cpp.FiniteDomain(dataset=precomputed_sample_df, target_column=TARGET_COLUMN)

###########################
# Starting up the MCMC...
###########################

# > Algorithm 1.1: Initial Stage: draw `I` initial samples from a latin hypercube design in `A` (domain)

# Draw initial points from domain (as np array)
initial_points_array = domain.SamplePointsInDomain(objective_func._num_init_pts)
# Evaluate function in initial points
initial_points_value = np.array([objective_func.evaluate(pt) for pt in initial_points_array])
# Build points using custom container class
initial_points = [data_containers.SamplePoint(pt,
                                              [initial_points_value[num, i]
                                               for i in objective_func.observations],
                                              objective_func._sample_var)
                  for num, pt in enumerate(initial_points_array)]

# Build historical data container
initial_data = data_containers.HistoricalData(dim=objective_func._dim,
                                              num_derivatives=objective_func.n_derivatives)

initial_data.append_sample_points(initial_points)

# initialize the model
prior = default_priors.DefaultPrior(1 + objective_func._dim + objective_func.n_observations, objective_func.n_observations)

# noisy = False means the underlying function being optimized is noise-free
gp_loglikelihood = log_likelihood_mcmc.GaussianProcessLogLikelihoodMCMC(
    historical_data=initial_data,
    derivatives=objective_func.derivatives,
    prior=prior,
    chain_length=1000,
    burnin_steps=2000,
    n_hypers=2 ** 4,  # 16 random walkers
    noisy=False
)
gp_loglikelihood.train()

###########################
# Main cycle
###########################

# Algorithm 1.2: Main Stage: For `s` to `N`
for s in range(N_ITERATIONS):
    print(f"{MODE}, "
          f"{s}th iteration, "
          f"func={objective_func_name}, "
          f"q={N_POINTS_PER_ITERATION}")
    time1 = time.time()

    if MODE == 'KG':

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
        pass

    else:  # method == 'EI':
        next_points, voi = bayesian_optimization.gen_sample_from_qei(
            gp_loglikelihood.models[0],
            domain,
            KG_gradient_descent_params,
            N_POINTS_PER_ITERATION,
            num_mc=2 ** 10
        )

    print(f"{MODE} takes {(time.time()-time1)} seconds")
    print("Suggests points:")
    print(next_points)



    # > ALgorithm 1.5: 5: Sample these points (z∗1 , z∗2 , · · · , z∗q)
    # > ...
    sampled_points = [data_containers.SamplePoint(pt,
                                                  objective_func.evaluate(pt)[objective_func.observations],
                                                  objective_func._sample_var)
                      for pt in next_points]

    time1 = time.time()

    # > ...
    # > re-train the hyperparameters of the GP by MLE
    # > and update the posterior distribution of f
    gp_loglikelihood.add_sampled_points(sampled_points)
    gp_loglikelihood.train()
    print(f"Retraining the model takes {time.time() - time1} seconds")
    time1 = time.time()

    # > Algorithm 1.7: Return the argmin of the average function `μ` currently estimated in `A`

    # In the current implementation, this argmin is found
    # and returned at every interaction (called "report_point").

    print("\nIteration finished successfully!")
    # print(f"The recommended point: {report_point}")
    # print(f"The recommended integer point: {np.round(report_point).astype(int)}")
    # print(f"Finding the recommended point takes {time.time()-time1} seconds")
    # print(f" {MODE}, VOI {voi}, best so far {objective_func.evaluate_true(report_point)[0]}")
