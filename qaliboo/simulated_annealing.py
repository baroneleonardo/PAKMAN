import numpy as np

from moe.optimal_learning.python.cpp_wrappers import knowledge_gradient_mcmc as KG

####################################
#####  SIMULATED ANNEALING  ########
####################################
# Implememntation of the simulated annealing alghoritm
# NB: it can be used insted/with the SGD to find the minima
#  
# Remember: kg have to be of the type KG imported above
#
# TEMPERATURE      APPROX.METHOD      MODE 
# high             random walk        global exploration
# med              sgd                improvement focused
# low              gd                 local search/exploit

def generate_neighbor_point(domain, current_point, max_relative_change):

    num_samples, num_features = current_point.shape
    
    new_points = current_point.copy()
    
    # TODO set this random vector proportional to the problem that I'm solving (ex: LiGen last feature)
    random_vectors = np.random.uniform(-max_relative_change, max_relative_change, size=(num_samples, num_features))

    for k in range(num_samples):
            new_point_update = domain.compute_update_restricted_to_domain(max_relative_change, new_points[k], random_vectors[k])
            new_points[k] = new_points[k] + new_point_update

    return new_points

def simulated_annealing(domain, kg, initial_point, num_iterations, initial_temperature, 
                        max_relative_change, typeT='log', alpha=1):
    
    current_point = initial_point
    kg.set_current_point(current_point)
    current_value = kg.compute_objective_function() # the same of compute_knoledge_gradient_mcmc()  

    for iteration in range(num_iterations):
        
        new_point = generate_neighbor_point(domain, current_point, max_relative_change)
        kg.set_current_point(new_point)
        new_value = kg.compute_objective_function()


        delta = new_value - current_value

        if delta < 0 or np.random.uniform(0, 1) < np.exp(-delta / temperature(iteration, initial_temperature, typeT, alpha)):
            current_point = new_point
            current_value = new_value

    return current_point

def temperature(iteration, initial_temperature, typeT, alpha): 
    if typeT=='log':
        return initial_temperature / (1+alpha*np.log(1 + iteration))
    elif typeT=='linear':
        if (initial_temperature - alpha*iteration)==0:
            return 1
        else:
            return initial_temperature - alpha*iteration
    elif typeT=='exp':
        return initial_temperature*alpha**(iteration+1)
    elif typeT=='quad':
        return initial_temperature/(1+alpha*iteration**2)
    else:
        raise KeyError("Insert a valid type for temperature")  