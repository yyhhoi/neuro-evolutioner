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

    return anatomy_matrix, labels_indexes, anatomy_matrix.shape[0]


def create_types_SG(num_neurons, anatomy_labels):
    types_matrix = np.random.uniform(0,1, (num_neurons, num_neurons))
    excite_inhib_sep = np.random.randn()
    types_matrix[types_matrix >= excite_inhib_sep] = 1
    types_matrix[types_matrix < excite_inhib_sep] = 0
    # connections from sensory to brain are all excitatory
    types_matrix[0:anatomy_labels['sensory2'], anatomy_labels['sensory2']:anatomy_labels['brain']:] = 1
    return types_matrix


def create_types_SMC(num_neurons, anatomy_labels):
    types_matrix = np.random.uniform(0,1, (num_neurons, num_neurons))
    excite_inhib_sep = np.random.randn()
    types_matrix[types_matrix >= excite_inhib_sep] = 1
    types_matrix[types_matrix < excite_inhib_sep] = 0
    # connections from sensory to brain are all excitatory
    types_matrix[0:anatomy_labels['sensory'], anatomy_labels['sensory']:anatomy_labels['brain']:] = 1

    return types_matrix