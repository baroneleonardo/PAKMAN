import numpy as np
from scipy.optimize import minimize
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

def generate_neighbor_point(domain, current_point, step):

    num_samples, num_features = current_point.shape
    
    new_points = current_point.copy()
    
    
    # TODO set this random vector proportional to the problem that I'm solving (ex: LiGen last feature)
    #random_vectors = np.random.uniform(-max_relative_change, max_relative_change, size=(num_samples, num_features))
    random_vectors = np.random.normal(loc=0, scale=1, size=(num_samples, num_features))
    random_vectors = random_vectors*step
    for k in range(num_samples):
            new_point_update = domain.compute_update_restricted_to_domain(1, new_points[k], random_vectors[k])
            new_points[k] = new_points[k] + new_point_update


    return new_points

def temperature(iteration, initial_temperature, typeT, alpha): 
    if typeT=='log':
        return initial_temperature / (1+alpha*np.log(1 + iteration))
    elif typeT=='linear':
        return initial_temperature / alpha*iteration + 1
    elif typeT=='exp':
        return initial_temperature*alpha**(iteration+1)
    elif typeT=='quad':
        return initial_temperature/(1+alpha*iteration**2)
    else:
        raise KeyError("Insert a valid type for temperature")  

def simulated_annealing(domain, kg, initial_point, num_iterations, initial_temperature, 
                        step, typeT='log', alpha=1):
    
    current_point = initial_point
    kg.set_current_point(current_point)
    current_value = kg.compute_objective_function() # the same of compute_knoledge_gradient_mcmc()  

    for iteration in range(num_iterations):
        
        new_point = generate_neighbor_point(domain, current_point, step)
        kg.set_current_point(new_point)
        new_value = kg.compute_objective_function()


        delta = new_value - current_value
        use_delta = False

        if use_delta==True:
            if delta < 0 or np.random.uniform(0, 1) < np.exp(-delta / temperature(iteration, initial_temperature, typeT, alpha)):
                current_point = new_point
                current_value = new_value
        else:
            if np.random.uniform(0, 1) < np.exp(-delta / temperature(iteration, initial_temperature, typeT, alpha)):
                current_point = new_point
                current_value = new_value

    return current_point

def simulated_annealing_ML(domain, kg, ml_model, initial_point, num_iterations, initial_temperature, 
                        step, typeT='log', alpha=1):
    
    current_point = initial_point
    kg.set_current_point(current_point)
    identity = ml_model.nascent_minima(current_point)*ml_model.exponential_penality(current_point)
    current_value = kg.compute_objective_function()*identity # the same of compute_knoledge_gradient_mcmc()  

    for iteration in range(num_iterations):
        
        new_point = generate_neighbor_point(domain, current_point, step)
        kg.set_current_point(new_point)
        identity = ml_model.nascent_minima(new_point)*ml_model.exponential_penality(new_point)
        new_value = kg.compute_objective_function()*identity


        delta = new_value - current_value
        
        if delta < 0 or np.random.uniform(0, 1) < np.exp(-delta / temperature(iteration, initial_temperature, typeT, alpha)):
            current_point = new_point
            current_value = new_value

    return current_point


def pso_multi_point(cost_func, num_points=4, dim=8, num_particles=30, max_iter=100, w=0.5, c1=1, c2=2):
    # Initialize particles and velocities for multiple points
    particles = np.random.uniform(-5.12, 5.12, (num_points, num_particles, dim))
    velocities = np.zeros((num_points, num_particles, dim))

    # Initialize the best positions and fitness values for multiple points
    best_positions = np.copy(particles)
    best_fitness = np.array([[cost_func(p) for p in particle_set] for particle_set in particles])
    
    # Find the best overall position and fitness
    swarm_best_position = particles[np.unravel_index(np.argmin(best_fitness), best_fitness.shape)]
    swarm_best_fitness = np.min(best_fitness)

    # Iterate through the specified number of iterations, updating particles and velocities
    for i in range(max_iter):
        # Update velocities
        r1 = np.random.uniform(0, 1, (num_points, num_particles, dim))
        r2 = np.random.uniform(0, 1, (num_points, num_particles, dim))
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        # Update positions
        particles += velocities

        # Evaluate fitness of each particle set for each point
        fitness_values = np.array([[cost_func(p) for p in particle_set] for particle_set in particles])

        # Update best positions and fitness values
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        
        # Update overall best position and fitness
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.unravel_index(np.argmin(fitness_values), fitness_values.shape)]
            swarm_best_fitness = np.min(fitness_values)

    # Return the best solution found by the PSO algorithm
    return swarm_best_position, swarm_best_fitness
