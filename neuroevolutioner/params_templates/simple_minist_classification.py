import numpy as np

def create_anatomy(ana_partition_matrix, **kwargs):
    nums = [kwargs[key] for key in kwargs]
    labels = list(kwargs.keys())
    
    # Labels the indexes in anatomy_matrix with the associated areas.
    indexes = [0] + [sum(nums[:i+1]) for i in range(len(nums))]
    labels_indexes = dict()
    for i in range(len(labels)):
        labels_indexes[labels[i]] = indexes[i+1]
    
    # Map anatomy_matrix to partition matrix
    anatomy_matrix = np.zeros((max(indexes), max(indexes)))
    for i in range(len(indexes)):
        for j in range(len(indexes)):
            if i < len(indexes)-1 and j < len(indexes)-1:
                anatomy_matrix[indexes[i]:indexes[i+1], indexes[j]:indexes[j+1]] = ana_partition_matrix[i, j]

    return anatomy_matrix, labels_indexes

def create_types(num_neurons):
    types_matrix = np.random.uniform(0,1, (num_neurons, num_neurons))
    excite_inhib_sep = np.random.randn()
    types_matrix[types_matrix >= excite_inhib_sep] = 1
    types_matrix[types_matrix < excite_inhib_sep] = 0
    return types_matrix

ana_partition_matrix = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0]
])

num_neurons = 644 # 64 for sensory, 500 for brain, 
anatomy_matrix, anatomy_labels  = create_anatomy(ana_partition_matrix,     
                                                            sensory = 64,
                                                            brain = 500, 
                                                            class1 = 20, 
                                                            class2 = 20, 
                                                            class3 = 20, 
                                                            class4 = 20)
types_matrix = create_types(num_neurons)

ensemble_range = {
    "u": (-70e-3, 5e-3), # Normal dist (mean, std)
    "u_rest": (-70e-3, 5e-3), # Normal dist (mean, std)
    "r_m": (1e6, 1000e6), # log10uniform sampling
    "tau_m": (5e-3, 200e-3), # Uniform dist (low, high)
    "u_threshold": (-50e-3, -40e-3), # Uniform dist (low, high)
    "u_reset": (-46e-3, -60e-3), # Uniform dist (low, high)
    "sharpness": (2e-3, 0.5e-3), # Normal dist (mean, std)
    "tau_w": (30e-3 , 100e-3), # Uniform dist (low, high)
    "a": (-1e-9, 1e-9), # Uniform dist (low, high)
    "b": (5e-12, 60e-12) # Uniform dist (low, high)
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
    "tau_cons": (800, 1400), # Uniform
    "transmitter_constants": (1e-5, 1e-2), # Log10Uniform
    "tau_hom": (800, 1400), # Uniform
    "tau_ht": (10e-3, 1000e-3), # Log10Uniform 
    "eta": (1e-3, 5), # Log10Uniform
    "gamma": (0, 5), # Uniform
    "tau_H": (5, 20) # Uniform
    }

def SMC_template():
    return num_neurons, ensemble_range, syn_ranges, weights_ranges


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print(np.sum(anatomy_matrix))
    plt.imshow(anatomy_matrix)
    plt.show()