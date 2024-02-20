import numpy as np
from moe.optimal_learning.python.cpp_wrappers import knowledge_gradient_mcmc as KG
from scipy.optimize import minimize


# Implementation of the stochastic gradient ascent e multistart optimization of it
# NB for the multistrat funztion exist a better implementation in C++ called gen_sample_from_qkg_mcmc but you
# cannot access into the code since is wrapped in c++ and difficult to modify
# btw it's the same thing that use multistart function of this file (exept for the speed)

# Basic stocastic Gradient ascent
def sga_kg(kg, domain, new_point, para_sgd=100, 
           gamma=0.7, alpha=1.0, max_relative_change=0.5):
    
    n_samples, n_features = new_point.shape
    
    for j in range(para_sgd):

        alpha_t = alpha/((1+j)**gamma)     # otherwise alpha = alpha/(1+j)
        kg.set_current_point(new_point)

        G = kg.compute_grad_objective_function() # the same as compute_grad_knowledge_gradient_mcmc()
        G = alpha_t*G
        
        for k in range(n_samples):
            new_point_update = domain.compute_update_restricted_to_domain(max_relative_change, new_point[k], G[k])
            new_point[k] = new_point[k] + new_point_update
    
    return new_point             
        
# Stocastic gradient ascent with already implemented the multistart
def multistart_sga_kg(kg, domain, n_points, n_restarts, para_sgd=100, gamma=0.7, alpha=1.0, max_relative_change=0.5):
    report_point = []
    kg_list = []
    
    for r in range(n_restarts):

        init_point = np.array(domain.generate_uniform_random_points_in_domain(n_points))
        
        new_point = sga_kg(kg, domain, init_point, para_sgd, gamma, alpha, max_relative_change)

        report_point.append(new_point)
        kg.set_current_point(new_point)
        
        kg_list.append(kg.compute_objective_function())


    index = np.argmax(kg_list)
    best_point = report_point[index]

    return(best_point)

# Stocastic Gradient Ascent with projection penality 
def sga_kg_ml(kg, domain, new_point, ml_model, para_sgd=50, gamma=0.7, alpha=1.0, max_relative_change=1):
    n_samples, n_features = new_point.shape
    for j in range(para_sgd):

        alpha_t = alpha/((1+j)**gamma)     # otherwise alpha = alpha/(1+j)
        kg.set_current_point(new_point)

        G = kg.compute_grad_objective_function() # the same as compute_grad_knowledge_gradient_mcmc()
        G = alpha_t*G

        for k in range(n_samples):
            new_point_update = domain.compute_update_restricted_to_domain(max_relative_change, new_point[k], G[k])
            candidate = new_point[k] + new_point_update
            if not ml_model.check_inside([candidate]):
            
                candidate = adjust_to_satisfy_constraint(candidate, G[k], ml_model)
            new_point[k] = candidate

    
    return new_point             

def adjust_to_satisfy_constraint(point, grad, ml_model):
        for i in range(10):
            point -= 0.1*grad
            if ml_model.check_inside([point]):
                return point
        return point 
