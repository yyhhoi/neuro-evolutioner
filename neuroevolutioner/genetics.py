from .genetics_templates.basic_connectivity import BC_get_params_dict
from numpy.random import uniform, normal
import numpy as np
def log10uniform(low, high, size):
    return np.power(10, uniform(np.log10(low), np.log10(high), size = size))

def rectified_normal(mean, std, size):
    return np.maximum(0, normal(mean, std, size= size))

def sampling_for_initialisation(p1, p2, num_neurons, output_dimensions, method_num):
    functions = [normal, rectified_normal, uniform, log10uniform]
    sizes = [(num_neurons,), (num_neurons, num_neurons)]
    return functions[method_num](p1, p2, sizes[output_dimensions-1])


def initialise_BC_params():
    params_dict = BC_get_params_dict()
    num_neurons = params_dict["num_neurons"]
    non_sampling_keys = ["num_neurons", "anatomy_matrix", "anatomy_labels", "SBA_labels", "types_matrix"]
    configs = dict()
    for params_key in params_dict.keys():
        if params_key in non_sampling_keys:
            configs[params_key] = params_dict[params_key]
        else:
            p1, p2, sampling_method, dimension = params_dict[params_key]
            array = sampling_for_initialisation(p1, p2, num_neurons, dimension, sampling_method)
            configs[params_key] = array
    configs["E_syn"] = configs["E_syn_inhib"] * (1 - configs["types_matrix"]) + configs["E_syn_excite"] * configs["types_matrix"]
    return configs

if __name__ == "__main__":
    configs  = initialise_BC_params()
    print()


