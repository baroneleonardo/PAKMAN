import logging
import time
import numpy as np
import numpy.linalg
import random
from tqdm import tqdm 
from moe.optimal_learning.python import data_containers
from moe.optimal_learning.python.cpp_wrappers import log_likelihood_mcmc, optimization as cpp_optimization, knowledge_gradient
from moe.optimal_learning.python.python_version import optimization as py_optimization
from moe.optimal_learning.python import default_priors
from moe.optimal_learning.python import random_features
from moe.optimal_learning.python.cpp_wrappers import knowledge_gradient_mcmc as KG
from examples import  auxiliary
from qaliboo import SGA as sga
from qaliboo.machine_learning_models import ML_model
import multiprocessing
from qaliboo import aux
from qaliboo import simulated_annealing as SA 
from sklearn.metrics import mean_absolute_percentage_error as mape

logging.basicConfig(level=logging.NOTSET)
_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)

class ParallelMaliboo:
    def __init__(self, n_initial_points: int = 10, n_iterations: int = 30, batch_size:int = 4,
                 m_domain_discretization: int= 30, objective_func = None, domain=None, lb: float=None, 
                 ub: float=None, nm:bool=False, uniform_sample:bool=True, n_restarts:int = 15, save:bool=False):
        """
        Initializes an instance of ParallelMaliboo.

        Args:
            n_initial_points (int): Number of initial points.
            n_iterations (int): Number of iterations.
            batch_size (int): Batch size.
            m_domain_discretization (int): Domain discretization dimension.
            objective_func (callable): Objective function to be optimized.
            lb (float): Lower bound of the domain.
            ub (float): Upper bound of the domain.
            nm (bool): True if nascent minima penalization is used.
            uniform_sample (bool): True if domain is to be uniformly sampled (False if sample from the global optimum).
            n_restarts (int): Number of restarts for optimization.
            save (bool): True if the results have to be saved
        """
        self._n_initial_points = n_initial_points
        self._n_iterations = n_iterations
        self._q = batch_size
        self._m = m_domain_discretization
        self._domain = domain
        self._objective_func = objective_func
        self._ub = ub
        self._lb = lb
        self._nm = nm
        self._uniform_sample = uniform_sample
        self._n_restarts=n_restarts
        self._save=save
        
        self._py_sgd_params_ps = py_optimization.GradientDescentParameters(
            max_num_steps=1000, max_num_restarts=3,
            num_steps_averaged=15, gamma=0.7, pre_mult=1.0,
            max_relative_change=0.02, tolerance=1.0e-10)

        self._cpp_sgd_params_ps = cpp_optimization.GradientDescentParameters(
            num_multistarts=5, max_num_steps=6, max_num_restarts=3,
            num_steps_averaged=3, gamma=0.0, pre_mult=1.0,
            max_relative_change=0.2, tolerance=1.0e-10)

        self._global_time=0

        #initial_points_array = self._domain.generate_uniform_random_points_in_domain(n_initial_points)
        initial_points_array= self._domain.sample_points_in_domain(n_initial_points)

        initial_points_value, initial_points_index, initial_points_time = self.evaluate_next_points(initial_points_array)

        initial_points = [data_containers.SamplePoint(pt,
                                              initial_points_value[num])
                  for num, pt in enumerate(initial_points_array)]
        
        initial_data = data_containers.HistoricalData(dim=objective_func.dim)
        initial_data.append_sample_points(initial_points)

        self._min_evaluated = np.min(initial_points_value)

        # Define the Machine Learning Model
        self._use_ml=False
        if (ub is not None) or (lb is not None) or (nm is not False):
            self._use_ml = True
            _log.info("You have selected an acquisition function with ML integrated")
            self._ml_model = ML_model(X_data=initial_points_array, 
                        y_data=np.array([objective_func.evaluate_time(pt) for pt in initial_points_array]), 
                        X_ub=ub,
                        X_lb=lb) 
        else: _log.info("Without ML model")

        # Initialize the Gaussian Process
        self._gp_loglikelihood = log_likelihood_mcmc.GaussianProcessLogLikelihoodMCMC(
            historical_data=initial_data,
            derivatives=objective_func.derivatives,  # Questo valore quando passato è 0
            prior=default_priors.DefaultPrior(1 + objective_func.dim + objective_func.n_observations, objective_func.n_observations),
            chain_length=1000,
            burnin_steps=2000,
            n_hypers=1,
            noisy=True
        )
        self._gp_loglikelihood.train()

        
        if self._save:
            self._result_folder = aux.create_result_folder('StereoMatch_Async')
            aux.csv_init(self._result_folder, initial_points_index)
            aux.csv_history(self._result_folder,-1,initial_points_index)
    
    
    def sync_optimization(self):
        '''
        Syncronous Knoledge Gradient optimization.
        '''
        _log.info("PARALLEL SYNCRONOUS BAYESIAN OPTIMIZATION")
        for s in range(self._n_iterations):
            self.iteration_step(s)
            if self._objective_func.evaluation_count>1000:
                break

        _log.info("\nOptimization finished successfully")

    def iteration_step(self, s):
        '''
        Performs a single optimization iteration.
        '''
        _log.info(f"{s}th iteration,  "f"q={self._q}")
        init_alg_time = time.time()
        # Define acquisition function
        kg = self.acquisition_function(self._q)
        # Multistart optimization of the acquisition function
        next_points = self.multistart_optimization(kg, self._q)
        # Evaluation objective function 
        next_points_value, next_points_index, next_points_time = self.evaluate_next_points(next_points)        # using sequential approach
        #next_points_value, next_points_index, next_points_time = self.evaluate_parallel_next_points(next_points) # using multiprocessorr
       
        max_time = max(next_points_time)

        # Update the regression model and the gaussian process.
        target, mape_value = self.update_model(next_points, next_points_value, next_points_index, s)
        
        # Compute the minimum of the posterior
        suggested_minimum = self.find_suggested_minimum()
        
        # Compute Global time
        alg_time = time.time() - init_alg_time
        _log.info(f"Optimization algorithm takes {(alg_time)} seconds")
        _log.info(f"Evaluate the objective function takes {(max_time)} seconds")
        
        self._global_time += max_time + alg_time

        # Compute unfeasible points
        if self._use_ml: unfeasible_points = self._ml_model.out_count(target)
        else: unfeasible_points = 0

        self.log_iteration_result(suggested_minimum, s, self._q, unfeasible_points)

        if self._save:
            aux.csv_info(s,self._q, self._min_evaluated, self._objective_func.evaluation_count,
                         self._global_time, unfeasible_points, mape_value, self._result_folder)
    
    def acquisition_function(self, q, points_being_sampled=None):
        '''
        Definition of the acquisition function.
        '''
        cpp_gaussian_process = self._gp_loglikelihood.models[0]
        # Sampling of the domain discretization (by or not sampling from the global optima)
        discrete_pts_list = self.domain_sample(self._uniform_sample,cpp_gaussian_process)
        ps_evaluator = knowledge_gradient.PosteriorMean(self._gp_loglikelihood.models[0], 0)
        ps_sgd_optimizer = cpp_optimization.GradientDescentOptimizer(self._domain,ps_evaluator,self._cpp_sgd_params_ps)        
        
        kg = KG.KnowledgeGradientMCMC(gaussian_process_mcmc=self._gp_loglikelihood._gaussian_process_mcmc,
                                        gaussian_process_list=self._gp_loglikelihood.models,
                                        num_fidelity=0,
                                        inner_optimizer=ps_sgd_optimizer,
                                        discrete_pts_list=discrete_pts_list,
                                        num_to_sample=q,
                                        num_mc_iterations=2**7,
                                        points_being_sampled=points_being_sampled,
                                        points_to_sample=None)
        return kg

    def multistart_optimization(self, kg, q):
        '''
        Multistart Optimization.
        '''
        seeds = np.random.randint(0, 10000, size=self._n_restarts)
        report_point=[]
        kg_list = []
        for i in range(self._n_restarts):
            new_point, kg_value = self.optimize_point(seeds[i], kg, q)
            report_point.append(new_point)
            kg_list.append(kg_value)
        index = np.argmax(kg_list)
        next_points = report_point[index]
        return next_points
    
    def optimize_point(self, seed, kg, q):
        '''
        Gradient Ascent + Machine Learning Optimization.
        ''' 
        np.random.seed(seed)
        init_point = np.array(self._domain.generate_uniform_random_points_in_domain(q))
        # Stocastic Gradient Ascent
        
        if self._use_ml:
            new_point = SA.simulated_annealing_ML(self._domain, kg, self._ml_model, init_point, 40, 3, 0.1)
            new_point = sga.stochastic_gradient_ml(kg, self._domain, init_point, self._ml_model)
        else:
            new_point = SA.simulated_annealing(self._domain, kg, init_point, 40, 2, 0.1)
            new_point = sga.stochastic_gradient(kg, self._domain, init_point)
            
        kg.set_current_point(new_point)
        # Machine Learning Penalization method
        identity = 1
        if self._nm:    
            identity *= self._ml_model.nascent_minima(new_point)
        if self._ub is not None or self._lb is not None:
            identity*=self._ml_model.exponential_penality(new_point, 10)
        kg_value = kg.compute_knowledge_gradient_mcmc()*identity 
        return new_point, kg_value  


    def _evaluate_point(self, pt):
        result = self._objective_func.evaluate(pt)
        if isinstance(result, tuple): 
            poi_v, poi_i, poi_t = result 
            return poi_v, poi_i, np.array(poi_t)
        else:
            return result, None, None
    
    def _wrapper_func(self, pt, queue):
        result = self._evaluate_point(pt)
        '''
        if result[2] is not None:
            print(f'Waiting time : {result[2]/self._time_proportion}')
            time.sleep(result[2]/self._time_proportion)
        '''
        queue.put(result)
        return
    
    def evaluate_next_points(self, next_points):
        '''
        Evaluate objective function at the next points in sequential order (just one process).
        '''
        dim = len(next_points)
        next_points_value = np.zeros(dim)
        next_points_index = np.zeros(dim)
        next_points_time = np.zeros(dim)
        for count, pt in enumerate(next_points):
            poi_v, poi_i, poi_t = self._evaluate_point(pt)
            next_points_value[count] = poi_v
            next_points_index[count] = poi_i
            next_points_time[count] = poi_t
        return next_points_value, next_points_index, next_points_time
    
    def evaluate_parallel_next_points(self, next_points):
        '''
        Evaluate objective function at the next points using process in parallel.
        '''
        results = []
        queue = multiprocessing.Queue()
        processes = []
        for pt in next_points:
            process = multiprocessing.Process(target=self._wrapper_func, args=(pt, queue))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
        while not queue.empty():
            results.append(queue.get())
        self._objective_func.add_evaluation_count(self._q)
        next_points_value, next_points_index, next_points_time = zip(*results)
        return np.array(next_points_value), np.array(next_points_index), np.array(next_points_time)
    
    def update_model(self, next_points, next_points_value, next_points_index, s):
        '''
        Update the regression model and the gaussian process.
        '''
        self._min_evaluated = min([self._min_evaluated, np.min(next_points_value)])
        sampled_points = [data_containers.SamplePoint(pt, next_points_value[num])
                            for num, pt in enumerate(next_points)]
        # Save the data
        if self._save:
            aux.csv_history(self._result_folder,s,next_points_index)

        # Update the ML model
        if self._use_ml:
            target = np.array([self._objective_func.evaluate_time(pt) for pt in next_points])
            # Compute the Mean Absolute Percentage Error (MAPE) 
            predictions = self._ml_model.predict(next_points)
            mape_value = mape(target, predictions)
            self._ml_model.update(next_points, target)
        else: 
            target = None
            mape_value = 0


        # Update the Gaussian Process 
        self.update_gp_loglikelihood(sampled_points)

        return target, mape_value


    def update_gp_loglikelihood(self, sampled_points):
        '''
        Update of the Gaussian Process with the new points sampled.
        '''
        self._gp_loglikelihood.add_sampled_points(sampled_points)
        self._gp_loglikelihood.train()
        return
    
    def find_suggested_minimum(self):
        '''
        Compute minimum of the posterior distribution.
        '''
        suggested_minimum = auxiliary.compute_suggested_minimum(self._domain, self._gp_loglikelihood, self._py_sgd_params_ps)
        _, _, closest_point_in_domain = self._domain.find_distance_index_closest_point(suggested_minimum)
        computed_cost = self._objective_func.evaluate(closest_point_in_domain, do_not_count=True)[0]
        return computed_cost

    def domain_sample(self, uniform, cpp_gaussian_process):
        '''
        Domain discretization.
        '''
        discrete_pts_list = []
        if uniform:
            eval_pts = self.unifrom_domain_sample(cpp_gaussian_process)
        else: eval_pts = self.global_optimum_sample(cpp_gaussian_process)
        discrete_pts_list.append(eval_pts)
        return discrete_pts_list
    
    def unifrom_domain_sample(self, cpp_gaussian_process):
        '''
        Domain discretization with uniform sample.
        '''
        eval_pts = self._domain.generate_uniform_random_points_in_domain(int(self._m))  # Sample continuous
        eval_pts = np.reshape(np.append(eval_pts,
                                        (cpp_gaussian_process.get_historical_data_copy()).points_sampled[:, :(self._gp_loglikelihood.dim)]),
                            (eval_pts.shape[0] + cpp_gaussian_process.num_sampled, cpp_gaussian_process.dim))
        return eval_pts
    
    def global_optimum_sample(self, cpp_gaussian_process):
        '''
        Domain discretization with sample from the global optimum.
        '''
        init_points = self._domain.generate_uniform_random_points_in_domain(int(1e2))
        discrete_pts_optima = random_features.sample_from_global_optima(cpp_gaussian_process,100, 
                                                                        self._objective_func.search_domain,init_points,self._m)
        eval_pts = np.reshape(np.append(discrete_pts_optima,
                                        (cpp_gaussian_process.get_historical_data_copy()).points_sampled[:, :(self._gp_loglikelihood.dim)]),
                                        (discrete_pts_optima.shape[0] + cpp_gaussian_process.num_sampled, cpp_gaussian_process.dim))
        return eval_pts

    
    def log_iteration_result(self, computed_cost, s, dimension, unfeasible, map_value=0):
        '''
        Logs information about the completed iteration.
        '''
        _log.info(f"""

        {s} - Iteration finished successfully!

        Points evaluated: {dimension}
        Minimum of the posterior: {computed_cost}
        \033[91mMinimum evaluated: {self._min_evaluated}\033[0m
        Objective function evaluations: {self._objective_func.evaluation_count}
        \033[92mUnfeasible points: {unfeasible}\033[0m
        Error: {np.linalg.norm(self._objective_func.min_value - computed_cost)}
        Error ratio: {np.abs(np.linalg.norm(self._objective_func.min_value - computed_cost) / self._objective_func.min_value)}
        \033[93mOptimizer time: {self._global_time}\033[0m
        MAPE: {map_value}
        """)

    # Definisci la tua funzione func_obj per valutare i punti (Inglobala con l'altra)
    def func_obj(self, point, queue):
        '''
        Fake simulation of the objective function
        '''
        result = self._objective_func.evaluate(point)
        if isinstance(result, tuple):
            poi_v, poi_i, poi_t = result
            fake_time = poi_t/self._time_proportion
            time.sleep(fake_time)
            queue.put((point, poi_v, poi_i))  
        else: queue.put((point, poi_v, None))
        return 
    
    def async_optimization(self, t_restart, n_process):
        '''
        Asyncronous Optimization.

            Args.
            t_restarts: waiting time before compute a new optimization (iter).
            n_process: number of parallelism of the algorithm.
        '''
        _log.info("PARALLEL ASYNCRONOUS BAYESIAN OPTIMIZATION")
        queue = multiprocessing.Queue() # Create a common queue to all process
        active_process = []
        results = []
        assigned_points = {}
        points_in_process = None
        s = 0 # Number of iteration
        self._time_proportion = 250 # Constant for proportional time #5 in Ligen
        #self._time_proportion = 5 # Constant for Ligen
        #self._time_proportion = 
        time0 = time.time()
        #self._time_proportion = 250 # COnstant for StereoMatch
        while True:
            
            time1 = time.time()
            # Avvio di nuovi processi se necessario
            if len(active_process) < n_process:
                q = n_process - len(active_process)
                _log.info(f"q = {q}")
                # Acquisition function optimization
                kg = self.acquisition_function(q, points_in_process)
                points_to_explore = self.multistart_optimization(kg, q)
                
                # Deliver points where compute the objective function to the processes
                for point in points_to_explore:
                    proc = multiprocessing.Process(target=self.func_obj, args=(point , queue))
                    proc.start()
                    active_process.append(proc)
                    assigned_points[proc] = point
            
            # Simulate waiting time
            if t_restart > 0:
                for _ in tqdm(range(t_restart), desc="Wait", unit="second"):
                    time.sleep(1)
                #time.sleep(t_restart)
          
            for proc in active_process:
                if not proc.is_alive(): # Check the terminated process
                    res = queue.get()  # Save the result
                    if res is not None:
                        results.append(res)
                    del assigned_points[proc]   # Delete the inactive processes 
                    active_process.remove(proc) 
            
            # Get the points that are still in process
            points_in_process = [assigned_points[proc] for proc in active_process]
            
            # Update the model with the computed results
            if results:
                next_points = [[*res[0]] for res in results]
                next_points_value = []
                next_points_index = []

                for res in results:
                    next_points_value.append(res[1])
                    next_points_index.append(res[2])

                dimension = len(next_points_value)
                self._objective_func.add_evaluation_count(dimension) # Add evaluation count to the model
                # Update the model
                target, mape_value = self.update_model(next_points, next_points_value, next_points_index, s)

                # Compute the minimum of the posterior distribution
                suggested_minimum = self.find_suggested_minimum()

                #self._global_time += time.time() - time1 + t_restart*(self._time_proportion-1) # real time
                if t_restart > 0:
                    self._global_time += 60 + t_restart*(self._time_proportion)
                    #self._global_time += time.time() - time1 + t_restart*(self._time_proportion-1) # real time
                else:
                    self._global_time = (time.time() - time0)*self._time_proportion 

                # Compute unfeasible points
                if self._use_ml: unfeasible_points = self._ml_model.out_count(target)
                else: unfeasible_points = 0

                # Logging the results of the iteration 
                self.log_iteration_result(suggested_minimum, s, dimension, unfeasible_points, mape_value)
                
                # Save teh results
                if self._save:
                    aux.csv_info(s,dimension, self._min_evaluated, self._objective_func.evaluation_count,
                                self._global_time, unfeasible_points, mape_value, self._result_folder) # add mape
                results = [] # Reset the results 
                s+=1  
                
                _log.info("Iteration finished succesfully")
        

            if self._global_time > 300000:
                _log.info(f"Global time reached. Optimization finished succesfully!")
                break
    
