import numpy as np
from .template_creation_utils.utils import create_anatomy, create_types_DA


# Goals:
# 1. Sensory fires
# 2. Firing activities reverberate in Bain
# 3. Only after a while, the Action fires,
# such that activation of Action is delayed simply by network activity reverberation


ana_partition_matrix = np.array([
    [0, 1, 0],
    [0, 1, 1],
    [0, 0, 0]
])

anatomy_matrix, anatomy_labels, num_neurons  = create_anatomy(ana_partition_matrix,     
                                                            sensory1 = 1,
                                                            brain = 8,
                                                            action1 = 1)
# Dummy matrix. No use
SBA_partition_matrix = np.ones((num_neurons, num_neurons))

# It is to identify the labels for Sensory (S), Brain (B) and Action (A) as a unified paradigm for 
# calculation for "within-group" parameter distribution for gene inheritance.
_, SBA_labels, num_neurons  = create_anatomy(ana_partition_matrix,     
                                                            sensory = 1,
                                                            brain = 8, 
                                                            action = 1)


# Define distribution of the parameters
# (param1, param2, sampling_methods, dimension)
# sampling_methods = {0, 1, 2, 3}, 0 = Normal, 1 = rectified normal, 2 = Uniform, 3 = Log10Uniform
# (param1, param2) = (mean, std) in Normal distribution. While, (param1, param2) = (min, max) in Uniform/Log10Uniform distribution
# dimension = {1, 2}. 1 = 1d-array, 2 = 2d-array

params_dict = {
    # Ensemble
    "u_rest": (-70e-3, 5e-3, 0, 1), # Normal
    "r_m": (1e3, 1e8, 3, 1), # Log10Uniform 
    "tau_m": (1e-4, 1e-1, 3, 1), # Log10Uniform
    "u_threshold": (-50e-3, -40e-3, 2, 1), # Uniform
    "u_reset": (-70e-3, -60e-3, 2, 1), # Uniform 
    "sharpness": (2e-3, 0.5e-3, 1, 1), # Rect Normal
    "tau_w": (1 , 1e3, 3, 1), # Log10Uniform
    "a": (0, 0.5, 0, 1), # Normal
    "b": (1e-3, 100, 3, 1), # Log10Uniform  

    # Synaptic current
    "tau_syn": (1e-3, 1e-1, 3, 2), # Log10Uniform
    "E_syn_inhib": (-90e-3, -60e-3, 2, 2), # Uniform  
    "E_syn_excite": (-20, 30, 2, 2), # Uniform  
    "g_syn_constant": (2, 0.5, 1, 2), # Rect Normal

    # Weights
    "w": (0.3, 0.3, 1, 2), # Rect Normal
    "w_max": (1, 10, 2, 2), # Uniform  
    "tau_LTP": (1e-3 , 1, 3, 1), # Log10Uniform
    "tau_LTD": (1e-3 , 1, 3, 1), # Log10Uniform
    "tau_LTP_slow": (5e-2, 5, 3, 1), # Log10Uniform
    "A": (1e-5, 1e2, 3, 2), # Log10Uniform
    "beta": (1e-2, 1e2, 3, 2), # Log10Uniform
    "w_p": (0.5, 0.3, 1, 2), # Rect Normal
    "P": (1e-2, 1e2, 3, 2), # Log10Uniform
    "tau_cons": (1, 1e3, 3, 2), # Log10Uniform
    "transmitter_constants": (1e-5, 1, 3, 2), # Log10Uniform
    "tau_hom": (1, 2e3, 3, 2), # Log10Uniform, recommended 1200 (20mins)
    "tau_ht": (1e-1, 1e2, 3, 1), # Log10Uniform 
    "eta": (1e-5, 1e3, 3, 2), # Log10Uniform
    "gamma": (1e-5, 1e3, 3, 2), # Log10Uniform
    "tau_H": (1e-3, 1e3, 3, 2), # Log10Uniform
    
    # Other parameters with different structures
    "num_neurons": num_neurons,
    "anatomy_matrix": anatomy_matrix,
    "anatomy_labels": anatomy_labels,
    "SBA_labels": SBA_labels
    }

# Initialisation and gene's distribution inheritance should be separated

def DA_get_params_dict():
    params_dict["types_matrix"] = create_types_DA(num_neurons, anatomy_labels)
    return params_dict

if __name__ == "__main__":
    pass
    # for i in range(10):
    #     # pa = create_types_DA(num_neurons, anatomy_labels)
    #     # print(pa)
    #     pa = DA_get_params_dict()
    #     print(pa["types_matrix"])