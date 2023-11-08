import numpy as np

# Implementation of the stochastic gradient ascent e multistart optimization of it
# NB for the multistrat funztion exist a better implementation in C++ called gen_sample_from_qkg_mcmc but you
# cannot access into the code since is wrapped in c++ and difficult to modify
# btw it's the same thing that use multistart function of this file (exept for the speed)

def sga_kg(kg, domain, new_point, para_sgd=100, gamma=0.7, alpha=1.0, max_relative_change=0.5):
    n_samples, n_features = new_point.shape
    for j in range(para_sgd):

        alpha_t = alpha/((1+j)**gamma)     # otherwise alpha = alpha/(1+j)
        kg.set_current_point(new_point)

        G = kg.compute_grad_knowledge_gradient_mcmc()
        G = alpha_t*G
        
        for k in range(n_samples):
            new_point_update = domain.compute_update_restricted_to_domain(max_relative_change, new_point[k], G[k])
            new_point[k] = new_point[k] + new_point_update
    
    return new_point             
        

def multistart_sga_kg(kg, domain, n_points, n_restarts, para_sgd=100, gamma=0.7, alpha=1.0, max_relative_change=0.5):
    report_point = []
    kg_list = []
    
    for r in range(n_restarts):

        init_point = np.array(domain.generate_uniform_random_points_in_domain(n_points))
        
        new_point = sga_kg(kg, domain, init_point, para_sgd, gamma, alpha, max_relative_change)

        report_point.append(new_point)
        kg.set_current_point(new_point)
        
        kg_list.append(kg.compute_knowledge_gradient_mcmc())


    index = np.argmax(kg_list)
    best_point = report_point[index]

    return(best_point)

