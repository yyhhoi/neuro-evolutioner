import numpy as np
from .template_creation_utils.utils import create_anatomy, create_types_BC


# Intended Paradigms:
# 1. Sensory connects to Brain (Sensory fires, then Brain fires)
# 2. Brain connects to Action (Brain fires, then Action fires)
# 3. Action does not connects the Sensory (Action fires, then Sensory does not fire. True in reverse direction)

# If achieved, the next goal will be:
# Inherit the parameters from basic_connectivity(BC) experiment, show that the low connectivity holds in a large proportion of species under mutation methods
# Based on the low connectvity, show that Sensory-Action connection can be learnt through the associative region of the brain
# Precisely, the paradigm could be: 
#   1. Stimulate Sensory and Action together (training)
#   2. After training, when either Sensory or Action is stimulated, the another will fire (connection is grown by training)

# Define anatomy
# 0 for connection possible, 1 for no connection
ana_partition_matrix = np.array([
    [1, 1, 0],
    [0, 1, 1],
    [0, 1, 1]
])

anatomy_matrix, anatomy_labels, num_neurons  = create_anatomy(ana_partition_matrix,     
                                                            sensory1 = 5,
                                                            brain = 10, 
                                                            action1 = 5)
# Dummy matrix. No use
SBA_partition_matrix = np.ones((num_neurons, num_neurons))

# It is to identify the labels for Sensory (S), Brain (B) and Action (A) as a unified paradigm for 
# calculation for "within-group" parameter distribution for gene inheritance.
_, SBA_labels, num_neurons  = create_anatomy(ana_partition_matrix,     
                                                            sensory = 5,
                                                            brain = 10, 
                                                            action = 5)
types_matrix = create_types_BC(num_neurons, anatomy_labels)


# Define distribution of the parameters
# (param1, param2, sampling_methods)
# sampling_methods = {0, 1, 2}, 0 = Normal, 1 = rectified normal, 2 = Uniform, 3 = Log10Uniform
# (param1, param2) = (mean, std) in Normal distribution. While, (param1, param2) = (min, max) in Uniform/Log10Uniform distribution
# dimension = {1, 2}. 1 = 1d-array, 2 = 2d-array

params_dict = {
    # Ensemble
    "u_rest": (-70e-3, 5e-3, 0, 1), # Normal
    "r_m": (0.1e6, 1e6, 3, 1), # Log10Uniform 
    "tau_m": (1e-4, 1e2, 3, 1), # Log10Uniform
    "u_threshold": (-50e-3, -40e-3, 2, 1), # Uniform
    "u_reset": (-46e-3, -60e-3, 2, 1), # Uniform 
    "sharpness": (2e-3, 0.5e-3, 1, 1), # Rect Normal
    "tau_w": (1e-4 , 1e2, 3, 1), # Log10Uniform
    "a": (1e-9, 1e9, 3, 1), # Log10Uniform  
    "b": (1e-12, 60e12, 3, 1), # Log10Uniform  

    # Synaptic current
    "tau_syn": (1e-4, 1e2, 3, 2), # Log10Uniform
    "E_syn_inhib": (-90e-3, -60e-3, 2, 2), # Uniform  
    "E_syn_excite": (-20, 30, 2, 2), # Uniform  
    "g_syn_constant": (4e-13, 4e4, 3, 2), # Log10Uniform

    # Weights
    "w": (0.3, 0.3, 1, 2), # Rect Normal
    "w_max": (1, 10, 2, 2), # Uniform  
    "tau_LTP": (1e-4 , 1e2, 3, 1), # Log10Uniform
    "tau_LTD": (1e-4 , 1e2, 3, 1), # Log10Uniform
    "tau_LTP_slow": (5e-2, 5e2, 3, 1), # Log10Uniform
    "A": (1e-5, 1e5, 3, 2), # Log10Uniform
    "beta": (1e-5, 1e5, 3, 2), # Log10Uniform
    "w_p": (0.5, 0.3, 1, 2), # Rect Normal
    "P": (1e-5, 1e5, 3, 2), # Log10Uniform
    "tau_cons": (1, 1e3, 3, 2), # Log10Uniform
    "transmitter_constants": (1e-5, 1, 3, 2), # Log10Uniform
    "tau_hom": (1, 1000, 3, 2), # Log10Uniform, recommended 1200 (20mins)
    "tau_ht": (1e-5, 1e2, 3, 1), # Log10Uniform 
    "eta": (1e-5, 1e3, 3, 2), # Log10Uniform
    "gamma": (1e-5, 1e3, 3, 2), # Log10Uniform
    "tau_H": (1e-3, 1e3, 3, 2), # Log10Uniform
    
    # Other parameters with different structures
    "num_neurons": num_neurons,
    "anatomy_matrix": anatomy_matrix,
    "anatomy_labels": anatomy_labels,
    "SBA_labels": SBA_labels,
    "types_matrix": types_matrix
    }

# Initialisation and gene's distribution inheritance should be separated

def BC_get_params_dict():
    return params_dict

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print(np.sum(anatomy_matrix))
    plt.imshow(anatomy_matrix)
    plt.show()