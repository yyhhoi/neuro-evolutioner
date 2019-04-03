import numpy as np
from .template_creation_utils.utils import create_anatomy, create_types_SG


# 0 for connection possible, 1 for no connection
ana_partition_matrix = np.array([
    [1, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 0, 1],

])


anatomy_matrix, anatomy_labels, num_neurons  = create_anatomy(ana_partition_matrix,     
                                                            sensory1 = 5,
                                                            sensory2 = 5,
                                                            brain = 70, 
                                                            class1 = 5, 
                                                            class2 = 5)
types_matrix = create_types_SG(num_neurons, anatomy_labels)

ensemble_range = {
    "u": (-70e-3, 5e-3), # Normal dist (mean, std)
    "u_rest": (-70e-3, 5e-3), # Normal dist (mean, std)
    "r_m": (0.1e6, 1e6), # log10uniform sampling
    "tau_m": (1e-3, 500e-3), # Uniform dist (low, high)
    "u_threshold": (-50e-3, -40e-3), # Uniform dist (low, high)
    "u_reset": (-46e-3, -60e-3), # Uniform dist (low, high)
    "sharpness": (2e-3, 0.5e-3), # Normal dist (mean, std)
    "tau_w": (1e-3 , 500e-3), # Uniform dist (low, high)
    "a": (-1e-9, 1e-9), # Uniform dist (low, high)
    "b": (1e-12, 60e-12) # Uniform dist (low, high)
}
syn_ranges = {
    "tau_syn": (5e-3, 0.5e-3), # Normal dist (mean, std)
    "E_syn_inhib": (-90e-3, -60e-3), # Uniform dist (low, high)
    "E_syn_excite": (-20, 30), # Uniform dist (low, high)
    "g_syn_constant": (4e-3, 4e-13) # log10uniform sampling
}

weights_ranges = {
    "anatomy": anatomy_matrix,
    "types": types_matrix,
    "w": (0.3, 0.3), # Normal dist (mean, std)
    "w_max": (1, 10), # Uniform dist (low, high)
    "tau_LTP": (5e-3, 200e-3), # uniform dist (low, high)
    "tau_LTD": (5e-3, 200e-3), # uniform dist (low, high)
    "tau_LTP_slow": (500e-3, 2000e-3), # Uniform dist (low, high)
    "A": (1e-3, 1), # Log10Uniform
    "beta": (1e-3, 1), # Log10Uniform
    "w_p": (0.5, 0.3), # Normal dist (mean, std)
    "P": (1e-3, 10), # Log10Uniform
    "tau_cons": (1, 1000), # Uniform, recommended 1200 (20mins)
    "transmitter_constants": (1e-5, 1e-2), # Log10Uniform
    "tau_hom": (1, 1000), # Uniform, recommended 1200 (20mins)
    "tau_ht": (10e-3, 1000e-3), # Log10Uniform 
    "eta": (1e-3, 5), # Log10Uniform
    "gamma": (0, 5), # Uniform
    "tau_H": (5, 20) # Uniform
    }

def SG_template():
    return num_neurons, ensemble_range, syn_ranges, weights_ranges

def SG_anatomy():
    return anatomy_matrix, anatomy_labels

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print(np.sum(anatomy_matrix))
    plt.imshow(anatomy_matrix)
    plt.show()