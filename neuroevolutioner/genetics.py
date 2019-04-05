from genetics_templates.basic_connectivity import BC_get_params_dict
from genetics_templates.time_learning import TL_get_params_dict
from numpy.random import uniform, normal
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
def log10uniform(low, high, size):
    return np.power(10, uniform(np.log10(low), np.log10(high), size = size))

def rectified_normal(mean, std, size):
    return np.maximum(0, normal(mean, std, size= size))

def sampling_for_initialisation(p1, p2, num_neurons, output_dimensions, method_num):
    functions = [normal, rectified_normal, uniform, log10uniform]
    sizes = [(num_neurons,), (num_neurons, num_neurons)]
    return functions[method_num](p1, p2, sizes[output_dimensions-1])

def crossover(chromosome1, chromosome2, p=0.2):
    """
    Args:
        chromosome1: (1d-ndarray) parameters in a 1d sequence form.
        chromosome2: same as chromosome1
        p: (float) probability of each element in chromosomem1 being replaced by the corresponding element in chromosome2
    Returns:
        new_chromosome1: (1d-ndarray) crossed-over chromosome with p chance that each element is replaced by chromosome2
        new_chromosome2: vice versus
    """
    assert chromosome1.shape == chromosome2.shape
    ran_vec = np.random.uniform(size=chromosome1.shape)
    ran_vec_less = ran_vec < p
    ran_vec_more = ran_vec >= p
    
    new_chromosome1 = chromosome1.copy()
    new_chromosome2 = chromosome2.copy()

    new_chromosome1[ran_vec_less] = chromosome2[ran_vec_less]
    new_chromosome2[ran_vec_more] = chromosome1[ran_vec_more]
    return new_chromosome1, new_chromosome2

def mutation(chromosome, fraction = 0.2):
    """
    Each element in chromosome mutates with probability following a truncated (between 0 and 1) normal distribution ~ N(0, 0.2)
    When an element mutates, its new value is sampled from a normal distribution ~ N(old_value, old_value*fraction/probability_of_mutation).
    Hence, the rarer the mutation, the new value will vary with larger range.

    Args:
        chromosome: (1d-ndarray) Parameters in 1d sequence form.
        fraction: (float) Fraction that controls the variance of mutation range
    Returns        
        new_chromosome: (1d-ndarray) New chromosome after mutation. Invalid values for the program could exist, but it is mutation :).
    """
    new_chromosome = chromosome.copy()
    m = chromosome.shape[0]
    prob_mutate = np.clip(normal(0, 0.2, m), 0, 1)
    prob_observed = uniform(0, 1, m)
    mutate_mask = prob_observed < prob_mutate
    new_chromosome[mutate_mask] = normal(new_chromosome[mutate_mask],
                                         new_chromosome[mutate_mask]*fraction/prob_mutate[mutate_mask])
    return new_chromosome

class ParamsInitialiser(ABC):
    @abstractmethod
    def __init__(self):
        self.params_dict = dict()
        self.non_sampling_keys = ["num_neurons", "anatomy_matrix", "anatomy_labels", "SBA_labels", "types_matrix"]
        self.non_mutating_keys = ["num_neurons", "anatomy_matrix", "anatomy_labels", "SBA_labels"]

    def sample_new_configs(self):
        num_neurons = self.params_dict["num_neurons"]
        self.configs = dict()
        for params_key in self.params_dict.keys():
            if params_key in self.non_sampling_keys:
                self.configs[params_key] = self.params_dict[params_key]
            else:
                p1, p2, sampling_method, dimension = self.params_dict[params_key]
                array = sampling_for_initialisation(p1, p2, num_neurons, dimension, sampling_method)
                self.configs[params_key] = array
        self.configs["E_syn"] = self.configs["E_syn_inhib"] * (1 - self.configs["types_matrix"]) + self.configs["E_syn_excite"] * self.configs["types_matrix"]
        return self.configs

    
    def configs2gene(self, configs):
        shapes_list = []
        keys_list = []
        separation_idx = [0]
        all_arrays_np = np.array([])
        genes_dict = {"gene":dict(), "others": dict()}
        for key in configs.keys():
            if key in self.non_mutating_keys:
                genes_dict["others"][key] = configs[key]
            else:
                shapes_list.append(configs[key].shape)
                keys_list.append(key)

                # get index
                arr_flatten = configs[key].flatten()
                flatten_len = arr_flatten.shape[0]

                current_index = separation_idx[-1] + flatten_len
                separation_idx.append(current_index)
                
                # append to numpy
                all_arrays_np = np.append(all_arrays_np, arr_flatten)
        
        genes_dict["gene"]["shapes"] = shapes_list
        genes_dict["gene"]["keys"] = keys_list
        genes_dict["gene"]["separation_idx"] = separation_idx
        genes_dict["gene"]["chromosome"] = all_arrays_np
        return genes_dict
    
    @staticmethod
    def gene2configs( genes_dict):
        configs = dict()

        length_list = len(genes_dict["gene"]["shapes"])
        for idx in range(length_list):
            arr_start_idx, arr_end_idx = genes_dict["gene"]["separation_idx"][idx], genes_dict["gene"]["separation_idx"][idx+1]
            arr_retrieved = genes_dict["gene"]["chromosome"][arr_start_idx: arr_end_idx].reshape(genes_dict["gene"]["shapes"][idx])
            configs[genes_dict["gene"]["keys"][idx]] = arr_retrieved
        for others_key in genes_dict["others"].keys():
            configs[others_key] = genes_dict["others"][others_key]
        return configs

    @staticmethod
    def test_if_two_configs_are_equal(configs1, configs2):
        for config_key in configs1.keys():
            try:
                assert np.array_equal(configs1[config_key], configs2[config_key])
            except:
                print("Not matched in {}".format(config_key))



class BC_ParamsInitialiser(ParamsInitialiser):
    def __init__(self):
        super(BC_ParamsInitialiser, self).__init__()
        self.params_dict = BC_get_params_dict()

class TL_ParamsInitialiser(ParamsInitialiser):
    def __init__(self):
        super(TL_ParamsInitialiser, self).__init__()
        self.params_dict = TL_get_params_dict()
        self.non_mutating_keys = ["num_neurons", "anatomy_matrix", "anatomy_labels", "SBA_labels"]

class ConfigsConverter(ParamsInitialiser):
    def __init__(self, configs):
        super(ConfigsConverter, self).__init__()
        

class TL_FitnessMeasurer():
    def __init__(self, activity):
        """
        Args:
            activity: Pandas dataframe with shape (times, num_neurons)
        """
        self.activity = activity
        self.fitness_score = None
    def build_score_criteria(self, encourage=None, penalise=None):
        if encourage is None:
            self.encourage = {
                "train_S": [2],
                "train_A": [11],
                "test_S": [2],
                "Test_A":[11],
            }
        else:
            self.encourage = encourage
        if penalise is None:
            self.penalise = {
                "train_ISI": [2, 11],
                "test_ISI": [2, 11]
            }
        else:
            self.penalise = penalise

    def calc_fitness(self):
        self.fitness_score = 0
        for condition_label in self.encourage.keys():
            mask = (self.activity["condition"] == condition_label)
            firing_ocurrence = self.activity[mask].iloc[:, self.encourage[condition_label]]
            while isinstance(firing_ocurrence , float) is False:
                firing_ocurrence = firing_ocurrence.mean()
            self.fitness_score += firing_ocurrence

        for condition_label in self.penalise.keys():
            mask = (self.activity["condition"] == condition_label)
            firing_ocurrence = self.activity[mask].iloc[:, self.penalise[condition_label]] * -2
            while isinstance(firing_ocurrence , float) is False:
                firing_ocurrence = firing_ocurrence.mean()
            self.fitness_score += firing_ocurrence
    
        return self.fitness_score






if __name__ == "__main__":
    # # Debugging
    # params_init = TL_ParamsInitialiser()
    # configs = params_init.sample_new_configs()
    # custom_init = Custom_ParamsInitialiser(configs)
    # gene_dict = custom_init.configs2gene(configs)
    # configs_recovered = custom_init.gene2configs(gene_dict)
    # custom_init.test_if_two_configs_are_equal(configs, configs_recovered)
    pass


