import numpy as np

from moe.optimal_learning.python.cpp_wrappers import knowledge_gradient_mcmc as KG

# Remember: kg have to be of the type KG imported above

def generate_neighbor_point(domain, current_point, max_relative_change):

    num_samples, num_features = current_point.shape
    
    new_points = current_point.copy()
    
    random_vectors = np.random.uniform(-max_relative_change, max_relative_change, size=(num_samples, num_features))

    for k in range(num_samples):
            new_point_update = domain.compute_update_restricted_to_domain(max_relative_change, new_points[k], random_vectors[k])
            new_points[k] = new_points[k] + new_point_update

    return new_points

def simulated_annealing(domain, kg, initial_point, num_iterations, initial_temperature, max_relative_change):
    current_point = initial_point
    kg.set_current_point(current_point)
    current_value = kg.compute_knowledge_gradient_mcmc()  

    for iteration in range(num_iterations):
        
        new_point = generate_neighbor_point(domain, current_point, max_relative_change)
        kg.set_current_point(new_point)
        new_value = kg.compute_knowledge_gradient_mcmc()


        delta = new_value - current_value

        if delta < 0 or np.random.uniform(0, 1) < np.exp(-delta / temperature_schedule(iteration, initial_temperature)):
            current_point = new_point
            current_value = new_value

    return current_point

def temperature_schedule(iteration, initial_temperature): 
    return initial_temperature / np.log(2 + iteration)
