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
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.cpp_wrappers.domain import TensorProductDomain as cppTensorProductDomain
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.cpp_wrappers import knowledge_gradient_mcmc as KG
from examples import bayesian_optimization, auxiliary, synthetic_functions
from qaliboo import precomputed_functions, finite_domain
from qaliboo import simulated_annealing as SA
from qaliboo import SGA as sga
from qaliboo.machine_learning_models import ML_model, ConstraintPenality
from concurrent.futures import ProcessPoolExecutor
from examples.RealProblem import xgboostopt 
from examples.VirtualSensor import VS

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
    'Ackley8',
    'Levy4',  # This function implementation is probably wrong
    'Rastrigin5',
    'Schwefel7','XGBoost','RandomForest','GradientBoosting','CIFRAR10','Iris','RF','XGB','Hartmann6','Ackley5','Ackley6','Ackley7',
    'IrisRF','IrisGB','Schwefel5','Dejong6','AxisParallel7'

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
parser.add_argument('--upper_bound', '-ub', help='Upper Bound (ML model)', type=float, default=None)
parser.add_argument('--lower_bound', '-lb', help='Lower Bound (ML model)', type=float, default=None)
parser.add_argument('--nascent_minima', '-nm', help='Nascent Minima term (ML model)', type=bool, default=False)
params = parser.parse_args()

objective_func_name = params.problem

if objective_func_name == 'ParabolicMinAtOrigin':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([0.0, 0.0])

elif objective_func_name == 'ParabolicMinAtTwoAndThree':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([2.0, 3.0])

elif objective_func_name == 'Hartmann3':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([0.114614, 0.555649, 0.852547])
elif objective_func_name == 'Hartmann6':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])

elif objective_func_name == 'Branin':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([3.14, 2.28])

elif objective_func_name=='Ackley8':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0])

elif objective_func_name=='Ackley5':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([0.0, 0.0,0.0, 0.0, 0.0])
    ground_t = np.array([1,1,1,1,1])

elif objective_func_name=='Ackley6':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([0.0, 0.0,0.0, 0.0, 0.0, 0.0])
elif objective_func_name=='Ackley7':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])

elif objective_func_name=='Dejong6':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([0.0, 0.0,0.0, 0.0, 0.0, 0.0])

elif objective_func_name=='AxisParallel7':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
    ground_t = np.array([1,1,1,1,1,1,1])

elif objective_func_name=='Ackley7':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])



elif objective_func_name == 'Levy4':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([1.0, 1.0, 1.0, 1.0])

elif objective_func_name == 'Rastrigin5':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

elif objective_func_name == 'Schwefel5':
    objective_func = getattr(synthetic_functions, params.problem)()
    known_minimum = np.array([420.9687,420.9687,420.9687,420.9687,420.9687])

elif objective_func_name == 'XGBoost':
    objective_func = getattr(xgboostopt, params.problem)()
    known_minimum = None
elif objective_func_name == 'RandomForest':
    objective_func = getattr(xgboostopt, params.problem)()
    known_minimum = None
elif objective_func_name == 'GradientBoosting':
    objective_func = getattr(xgboostopt, params.problem)()
    known_minimum = None
elif objective_func_name == 'Iris':
    objective_func = getattr(xgboostopt, params.problem)()
    known_minimum = None
elif objective_func_name == 'IrisRF':
    objective_func = getattr(xgboostopt, params.problem)()
    known_minimum = None
elif objective_func_name == 'IrisGB':
    objective_func = getattr(xgboostopt, params.problem)()
    known_minimum = None
elif objective_func_name == 'CIFRAR10':
    objective_func = getattr(xgboostopt, params.problem)()
    known_minimum = None
elif objective_func_name == 'RF':
    objective_func = getattr(VS, params.problem)()
    known_minimum = None
elif objective_func_name == 'XGB':
    objective_func = getattr(VS, params.problem)()
    known_minimum = None


n_initial_points = params.init
n_iterations = params.iter
n_points_per_iteration = params.points
m_domain_discretization_sample_size = params.sample_size
lb = params.lower_bound
ub = params.upper_bound
nm = params.nascent_minima
cpp_domain = cppTensorProductDomain([ClosedInterval(objective_func.search_domain[i, 0], objective_func.search_domain[i, 1])
                                                  for i in range(objective_func.search_domain.shape[0])])

domain = pythonTensorProductDomain([ClosedInterval(objective_func.search_domain[i, 0], objective_func.search_domain[i, 1])
                                                 for i in range(objective_func.search_domain.shape[0])])

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
    max_relative_change=0.2,
    tolerance=1.0e-10)

min_evaluated = None

################################
##### Initial samples ##########
################################
initial_points_array = domain.generate_uniform_random_points_in_domain(n_initial_points)
initial_points_value = np.array([objective_func.evaluate(pt) for pt in initial_points_array])

#norm_2 = np.sqrt(np.sum(initial_points_array ** 2, axis=1))    # Euclidean Norm

#norm_1 = np.sum(np.abs(initial_points_array), axis=1)  # Manatthan Norm
#inf_norm = np.max(np.abs(initial_points_array), axis=1)       # Infinity Norm

min_norm = np.min(np.abs(initial_points_array), axis=1)

initial_points = [data_containers.SamplePoint(pt,
                                              initial_points_value[num])
                  for num, pt in enumerate(initial_points_array)]

initial_data = data_containers.HistoricalData(dim=objective_func.dim)
initial_data.append_sample_points(initial_points)

min_evaluated = np.min(initial_points_value)


#################################
###### ML model init.############
#################################

use_ml = False
if (ub is not None) or (lb is not None) or (nm is not False):
    use_ml = True

if(use_ml==True):
    print("You have selected an acquisition function with ML integrated")
else:
    print("Without ML model")

if use_ml == True:
    #ml_model = ConstraintPenality(ub, lb)
    
    ml_model = ML_model(X_data=initial_points_array, 
                        y_data=min_norm, 
                        X_ub=ub,
                        X_lb=lb) # Set this value if you are intrested in I(T(X) < X_ub)
    
#################################
######## GP init. ###############
#################################

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

###########################
###### Def. minima ########
###########################
if known_minimum is not None:
    _log.info(f'The minimum in the domain is:\n{known_minimum}')

###########################
####### Main cycle ########
###########################

results = []
result_file = f'./results/VincoliAttivi/{objective_func_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H%M")}.json'

# Algorithm 1.2: Main Stage: For `s` to `N`
for s in range(n_iterations):
    _log.info(f"{s}th iteration, "
              f"func={objective_func_name}, "
              f"q={n_points_per_iteration}")
    time1 = time.time()

    cpp_gaussian_process = gp_loglikelihood.models[0]
    
    ##################################
    #### Def. of the space A #########
    ##################################
    discrete_pts_list = []
    glb_opt_smpl = True    # Set to true if you want a dynamic space
    # X(1:n) + z(1:q) + sample from the global optima of the posterior
    
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

     
    ps_evaluator = knowledge_gradient.PosteriorMean(gp_loglikelihood.models[0], 0)
    ps_sgd_optimizer = cpp_optimization.GradientDescentOptimizer(
        cpp_domain,
        ps_evaluator,
        cpp_sgd_params_ps
    )

    # Selection of the R restarting points    

    kg = KG.KnowledgeGradientMCMC(gaussian_process_mcmc=gp_loglikelihood._gaussian_process_mcmc,
                                    gaussian_process_list=gp_loglikelihood.models,
                                    num_fidelity=0,
                                    inner_optimizer=ps_sgd_optimizer,
                                    discrete_pts_list=discrete_pts_list,
                                    num_to_sample=n_points_per_iteration,
                                    num_mc_iterations=2**7,
                                    points_to_sample=None
                                    )
    

    ################################
    # Multistart SGA & SA parameters
    ################################
    para_sgd = 200
    alpha = 1
    gamma = 0.7
    num_restarts = 20
    max_relative_change = 0.9
    initial_temperature = 1
    n_iter_sa = 40   #40

    report_point = []
    kg_list = []
    
    
    use_SA = False

    def optimize_point(seed):

        np.random.seed(seed)
        init_point = np.array(domain.generate_uniform_random_points_in_domain(n_points_per_iteration))
        
        if use_SA == False:
            new_point=init_point
        else:
            new_point = SA.simulated_annealing(domain, kg, init_point, n_iter_sa, initial_temperature, 0.001)
                
        new_point = sga.stochastic_gradient(kg, domain, new_point)

        kg.set_current_point(new_point)


        identity = 1
        
        if nm==True:    
            identity = identity*ml_model.nascent_minima(new_point)
        
        if (ub is not None) or (lb is not None):
            identity=identity*ml_model.exponential_penality(new_point, k=7)
            
        kg_value = kg.compute_knowledge_gradient_mcmc()*identity 
        
        return new_point, kg_value


    seeds = np.random.randint(0, 10000, size=num_restarts)
    with ProcessPoolExecutor() as executor:                     #max_workers=5
        res = list(executor.map(optimize_point, seeds))
        

    report_point, kg_list = zip(*res)

    index = np.argmax(kg_list)
    next_points = report_point[index]
    
    #next_points = sga.multistart_sga_kg(kg, domain, n_points_per_iteration, num_restarts, para_sgd, gamma, alpha, max_relative_change)


    _log.info(f"Knowledge Gradient update takes {(time.time()-time1)} seconds")
    _log.info("Suggests points:")
    _log.info(next_points)

    # > ALgorithm 1.5: 5: Sample these points (z∗1 , z∗2 , · · · , z∗q)
    # > ...
    
    next_points_value = np.array([objective_func.evaluate(pt) for pt in next_points])
    
    sampled_points = [data_containers.SamplePoint(pt,
                                              next_points_value[num])
                  for num, pt in enumerate(next_points)]

    
    # UPDATE OF THE ML MODEL
    if use_ml==True:       
        #norm_2 = np.sqrt(np.sum(next_points ** 2, axis=1))    # Euclidean Norm
        #norm_1 = np.sum(np.abs(initial_points_array), axis=1)  # Manatthan Norm
        #inf_norm = np.max(np.abs(initial_points_array), axis=1)       # Infinity Norm
        min_norm = np.min(np.abs(initial_points_array), axis=1)
        ml_model.update(next_points, min_norm)
    
    
    min_evaluated = np.min([min_evaluated, np.min(next_points_value)])
    time1 = time.time()

    # UPDATE OF THE GP
    # > re-train the hyperparameters of the GP by MLE
    # > and update the posterior distribution of f
    gp_loglikelihood.add_sampled_points(sampled_points)
    gp_loglikelihood.train()
    _log.info(f"Retraining the model takes {time.time() - time1} seconds")
    time1 = time.time()

    
    _log.info("\nIteration finished successfully!")
    _log.info(f"\n Minimum in Domain:{objective_func.evaluate(ground_t, do_not_count=True)}")
    ####################
    # Suggested Minimum
    ####################
    
    # > Algorithm 1.7: Return the argmin of the average function `μ` currently estimated in `A`

    # -> Nearest point in domain 
    #suggested_minimum = auxiliary.compute_suggested_minimum(domain, gp_loglikelihood, py_sgd_params_ps)
    suggested_minimum = auxiliary.compute_suggested_minimum(domain, gp_loglikelihood, py_sgd_params_ps)
    
    computed_cost = objective_func.evaluate(suggested_minimum, do_not_count=True) 
    _log.info(f'The evaluated minimum is {min_evaluated}')               
    _log.info(f"The suggested minimum is:\n {suggested_minimum}")    
    _log.info(f"Which has a cost of:\n {computed_cost}")
    _log.info(f"Finding the suggested minimum takes {time.time() - time1} seconds")
    _log.info(f'The target function was evaluated {objective_func.evaluation_count} times')

    
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
            minimum_evaluated = min_evaluated.tolist(),
            suggested_minimum=suggested_minimum.tolist(),
            #known_minimum=known_minimum.tolist(),
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