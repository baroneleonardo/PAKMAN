import datetime
import logging
import time
import json
import argparse
import os
import numpy as np
import numpy.linalg
import random
from sklearn.metrics import mean_absolute_percentage_error as mape
from moe.optimal_learning.python import data_containers
from moe.optimal_learning.python.cpp_wrappers import log_likelihood_mcmc, optimization as cpp_optimization, knowledge_gradient
from moe.optimal_learning.python.python_version import optimization as py_optimization
from moe.optimal_learning.python import default_priors
from moe.optimal_learning.python import random_features
from moe.optimal_learning.python.cpp_wrappers import knowledge_gradient_mcmc as KG
from examples import bayesian_optimization, auxiliary, synthetic_functions
from qaliboo import precomputed_functions, finite_domain
from qaliboo import simulated_annealing as SA
from qaliboo import SGA as sga
from qaliboo.machine_learning_models import ML_model
from concurrent.futures import ProcessPoolExecutor
from qaliboo import aux
from tqdm import tqdm

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
    'Ackley5',
    'Levy4',  # This function implementation is probably wrong
    # Data sets
    'Query26',
    'LiGen',
    'StereoMatch',
    'LiGenTot',
    'ScaledLiGen',
    'ScaledLiGenTot',
    'ScaledStereoMatch',
    'ScaledQuery26',
    'Rastrigin9',
    'ScaledStereoMatch10'
]

# Set random seed 
np.random.seed(np.random.randint(0, 10000))
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
parser.add_argument('--domain_upper_bound', '-dub', help='Domain Upper Bound', type=float, default=None)
parser.add_argument('--lower_bound', '-lb', help='Lower Bound (ML model)', type=float, default=None)
parser.add_argument('--nascent_minima', '-nm', help='Nascent Minima term (ML model)', type=bool, default=False)
params = parser.parse_args()

objective_func_name = params.problem

objective_func = getattr(precomputed_functions, params.problem)
known_minimum = objective_func.minimum
domain = objective_func

n_initial_points = params.init
n_iterations = params.iter
n_points_per_iteration = params.points
m_domain_discretization_sample_size = params.sample_size
lb = params.lower_bound
ub = params.upper_bound
nm = params.nascent_minima
dub = params.domain_upper_bound

py_sgd_params_ps = py_optimization.GradientDescentParameters(
    max_num_steps=1000, max_num_restarts=3,
    num_steps_averaged=15, gamma=0.7, pre_mult=1.0,
    max_relative_change=0.02, tolerance=1.0e-10)

cpp_sgd_params_ps = cpp_optimization.GradientDescentParameters(
    num_multistarts=5, max_num_steps=6, max_num_restarts=3,
    num_steps_averaged=3, gamma=0.0, pre_mult=1.0,
    max_relative_change=0.2, tolerance=1.0e-10)

min_evaluated = None

# CREARE LA CARTELLA DOVE SALVARE I FILE
results=[]
h_p = []
i_p = []
global_time=0
dat = aux.define_dat(objective_func_name)

'''
main_folder = './results/'
sub_folder_time = 'Time/'
if not os.path.exists(main_folder):
    os.makedirs(main_folder)
folder_path_time = os.path.join(main_folder, sub_folder_time)
if not os.path.exists(folder_path_time):
    os.makedirs(folder_path_time)
now_dir = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
folder_path_now = os.path.join(folder_path_time, now_dir)
if not os.path.exists(folder_path_now):
    os.makedirs(folder_path_now)

result_folder = folder_path_now
result_file = os.path.join(result_folder, 'result_file.json')
'''

################################
##### Initial samples ##########
################################

initial_points_array = domain.sample_points_in_domain(n_initial_points)


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
# SALVO I MIEI PUNTI INIZIALI
init = os.path.join(result_folder, 'init.json')
for i in range(n_initial_points):
    i_p.append(
        dict(
            point=initial_points_array[i].tolist(),
            value=initial_points_value[i].tolist(),
            dat_index = initial_points_index[i].tolist(),
            time = initial_points_time[i].tolist()))
   
    h_p.append(
        dict(
            iter = -1,
            point=initial_points_array[i].tolist(),
            value=initial_points_value[i].tolist(),
            dat_index = initial_points_index[i].tolist(),
            time = initial_points_time[i].tolist()))
    
with open(init, 'w') as f:
    json.dump(i_p, f, indent=2)
'''

use_ml = False
if (ub is not None) or (lb is not None) or (nm is not False):
    use_ml = True

if(use_ml==True):
    print("You have selected an acquisition function with ML integrated")
else:
    print("Without ML model")

if use_ml == True:
    if dub is None: dub = ub
    ml_model = ML_model(X_data=initial_points_array, 
                        y_data=np.array([objective_func.evaluate_time(pt) for pt in initial_points_array]), 
                        X_ub=dub,
                        X_lb=lb) # Set this value if you are intrested in I(T(X) < X_ub)

if nm == True: word = 'a3'
else: word = 'a2'
result_folder = aux.create_result_folder(f'Query_sync_{dub/1000}_{word}')


aux.csv_init(result_folder,initial_points_index, dat)
aux.csv_history(result_folder,-1,initial_points_index, dat)


initial_points = [data_containers.SamplePoint(pt,
                                              initial_points_value[num])
                  for num, pt in enumerate(initial_points_array)]
initial_data = data_containers.HistoricalData(dim=objective_func.dim)
initial_data.append_sample_points(initial_points)

min_evaluated = np.min(initial_points_value)





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


for s in range(n_iterations):
    _log.info(f"{s}th iteration, "
              f"func={objective_func_name}, "
              f"q={n_points_per_iteration}")
    init_alg_time = time.time()
    time1 = time.time()

    cpp_gaussian_process = gp_loglikelihood.models[0]
    
    ##################################
    #### Def. of the space A #########
    ##################################
    discrete_pts_list = []
    glb_opt_smpl = False     # Set to true if you want a dynamic space
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
        domain,
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
    para_sgd = 100 
    alpha = 1
    gamma = 0.7
    num_restarts = 15
    max_relative_change = 0
    initial_temperature = 2
    n_iter_sa = 50   #40

    report_point = []
    kg_list = []
    
    
    if objective_func.evaluation_count - n_initial_points < 50:
            error = 1.5 - 0.01*(objective_func.evaluation_count - n_initial_points)
            #print(error)
    else:
        error = 1



    def optimize_point(seed):

        np.random.seed(seed)
        init_point = np.array(domain.generate_uniform_random_points_in_domain(n_points_per_iteration))
        new_point=init_point
        #new_point = SA.simulated_annealing_ML(domain, kg, ml_model,init_point, n_iter_sa, initial_temperature, 0.01)
        new_point = SA.simulated_annealing(domain, kg, init_point, n_iter_sa, initial_temperature, 0.001)
        
        new_point = sga.stochastic_gradient(kg, domain, new_point)

        kg.set_current_point(new_point)
        '''
        if objective_func.evaluation_count - n_initial_points < 50:
            error = 1.5 - 0.01*(objective_func.evaluation_count - n_initial_points)
            print(error)
        else:
            error = 1
        '''
        error = 1
        identity = 1

        if nm==True:    
            identity = identity*ml_model.nascent_minima(new_point, error)
    
        if (ub is not None) or (lb is not None):
            #identity=identity*ml_model.identity(new_point)
            identity=identity*ml_model.exponential_penality(new_point, 7, error)

        kg_value = kg.compute_knowledge_gradient_mcmc()*identity 
        
        return new_point, kg_value


    seeds = np.random.randint(0, 10000, size=num_restarts)
    with ProcessPoolExecutor() as executor:
        res = list(executor.map(optimize_point, seeds))
        

    report_point, kg_list = zip(*res)
    index = np.argmax(kg_list)
    next_points = report_point[index]
    
    _log.info(f"Knowledge Gradient update takes {(time.time()-time1)} seconds")
    #_log.info("Suggests points:")
    #_log.info(next_points)

    next_points_value = np.zeros(n_points_per_iteration)
    next_points_index = np.zeros(n_points_per_iteration)
    next_points_time = np.zeros(n_points_per_iteration)
    count=0
    for pt in next_points:
        poi_v, poi_i, poi_t = objective_func.evaluate(pt) 
        next_points_value[count]= poi_v
        next_points_index[count]=poi_i
        next_points_time[count] = np.array(poi_t)
        count= count+1




    max_time = max(next_points_time)
    '''
    hist = os.path.join(result_folder, 'hist.json')
    for i in range(n_points_per_iteration):
        h_p.append(
        dict(
            iter = s,
            point=next_points[i].tolist(),
            value=next_points_value[i].tolist(),
            dat_index = next_points_index[i].tolist(),
            time = next_points_time[i].tolist()))
    
    with open(hist, 'w') as f:
        json.dump(h_p, f, indent=2) 
    '''   
    aux.csv_history(result_folder,s,next_points_index, dat)
    
    sampled_points = [data_containers.SamplePoint(pt,
                                              next_points_value[num])
                  for num, pt in enumerate(next_points)]

    
    # UPDATE OF THE ML MODEL

    if use_ml==True:
        target = np.array([objective_func.evaluate_time(pt) for pt in next_points])
        predictions = ml_model.predict(next_points)
        mape_value = mape(target, predictions)
        ml_model.update(next_points, target)
    
    
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

    ####################
    # Suggested Minimum
    ####################
    
    # > Algorithm 1.7: Return the argmin of the average function `μ` currently estimated in `A`
    suggested_minimum = auxiliary.compute_suggested_minimum(domain, gp_loglikelihood, py_sgd_params_ps)
    
    # -> Nearest point in domain
    _, _, closest_point_in_domain = domain.find_distance_index_closest_point(suggested_minimum)
    computed_cost = objective_func.evaluate(closest_point_in_domain, do_not_count=True)[0]
                 
    
    _log.info(f"The closest point in the finite domain is:\n {closest_point_in_domain}")
    _log.info(f"Which has a cost of:\n {computed_cost}")
    _log.info(f"Cost of the minimum evaluated:\n {min_evaluated}")
    _log.info(f"Finding the suggested minimum takes {time.time() - time1} seconds")
    _log.info(f'The target function was evaluated {objective_func.evaluation_count} times')
    #_log.info(f'Unfeasable Point:{ml_model.out_count(target)}')
    
    alg_time = time.time() - init_alg_time 
    #error = np.linalg.norm(objective_func.min_value - computed_cost)
    #error_ratio = np.abs(error/objective_func.min_value)
    #_log.info(f'Error: {error}')
    #_log.info(f'Error ratio: {error_ratio}')
    '''
    if use_ml: unfeasible_points = ml_model.out_count(target)
    else: unfeasible_points = 0
    '''
    unfeasible_points = 0
    global_time = global_time + max_time + 25
    _log.info(f'Optimizer Time: {global_time}')

    #aux.save_execution_time([next_points_time], result_folder)
    aux.csv_info(s,n_points_per_iteration, objective_func.evaluation_count,
                global_time, unfeasible_points, mape_value, result_folder, error)

_log.info("\nOptimization finished successfully!")



'''
    def make_plot():
        
        knowledge_gradient_values = []
        PAKMAN_values = []
        num_points = 200
        x_values = np.linspace(domain.lower_bound[0], domain.upper_bound[0], num_points)
        
        y_values = np.linspace(domain.lower_bound[1], domain.upper_bound[1], num_points)
        total_iterations = num_points * num_points
        progress_bar = tqdm(total=total_iterations, desc="Progress")
        
        for x in x_values:
            for y in y_values:

                point = np.array([x, y])
                
                kg.set_current_point(point)
                knowledge_gradient = kg.compute_knowledge_gradient_mcmc()
                knowledge_gradient_values.append(knowledge_gradient)
                

                identity = ml_model.nascent_minima(point.reshape(1, -1))
                PAKMAN_values.append(knowledge_gradient*identity)
                progress_bar.update(1)


        knowledge_gradient_values = np.array(knowledge_gradient_values).reshape((num_points, num_points))
        PAKMAN_values = np.array(PAKMAN_values).reshape((num_points, num_points))
        X, Y = np.meshgrid(x_values, y_values)

        data = np.column_stack((X.flatten(), Y.flatten(), knowledge_gradient_values.flatten()))
        np.savetxt('knowledge_gradient.csv', data, delimiter=',', header='X,Y,Knowledge Gradient', comments='')
        data = np.column_stack((X.flatten(), Y.flatten(), PAKMAN_values.flatten()))
        np.savetxt('PAKMAN.csv', data, delimiter=',', header='X,Y,PAKMAN', comments='')



'''