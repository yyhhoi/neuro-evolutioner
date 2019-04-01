import numpy as np
from numpy.random import uniform, normal
from params_templates.simple_minist_classification import SMC_template

def log10uniform(low, high, size):
    return np.power(10, uniform(np.log10(low), np.log10(high), size = size))

def rectified_normal(mean, std, size):
    return np.maximum(0, normal(mean, std, size= size))

def gen_params_dict(num_neurons, 
                    ensemble_range_dict,
                    syn_range_dict,
                    weights_ranges_dict
                    ):

    ensemble_dict = dict()
    ensemble_dict["u"] = rectified_normal(ensemble_range_dict["u"][0], ensemble_range_dict["u"][1], size= (num_neurons,))
    ensemble_dict["u_rest"] = rectified_normal(ensemble_range_dict["u_rest"][0], ensemble_range_dict["u_rest"][1], size= (num_neurons,))
    ensemble_dict["r_m"] = log10uniform(ensemble_range_dict["r_m"][0], ensemble_range_dict["r_m"][1], (num_neurons,))
    ensemble_dict["tau_m"] = uniform(ensemble_range_dict["tau_m"][0], ensemble_range_dict["tau_m"][1], size = (num_neurons,) )
    ensemble_dict["u_threshold"] = uniform(ensemble_range_dict["u_threshold"][0], ensemble_range_dict["u_threshold"][1], size = (num_neurons,) )
    ensemble_dict["u_reset"] = uniform(ensemble_range_dict["u_reset"][0], ensemble_range_dict["u_reset"][1], size = (num_neurons,) )
    ensemble_dict["sharpness"] = rectified_normal(ensemble_range_dict["sharpness"][0], ensemble_range_dict["sharpness"][1], size= (num_neurons,))
    ensemble_dict["tau_w"] = uniform(ensemble_range_dict["tau_w"][0], ensemble_range_dict["tau_w"][1], size = (num_neurons,) )
    ensemble_dict["a"] = uniform(ensemble_range_dict["a"][0], ensemble_range_dict["a"][1], size = (num_neurons,) )
    ensemble_dict["b"] = uniform(ensemble_range_dict["b"][0], ensemble_range_dict["b"][1], size = (num_neurons,) )

    syn_dict = dict()
    syn_dict["tau_syn"] = rectified_normal(syn_range_dict["tau_syn"][0], syn_range_dict["tau_syn"][1], size= (num_neurons,num_neurons))
    E_syn_inhib = uniform(syn_range_dict["E_syn_inhib"][0], syn_range_dict["E_syn_inhib"][1], size = (num_neurons,num_neurons) )
    E_syn_excite = uniform(syn_range_dict["E_syn_excite"][0], syn_range_dict["E_syn_excite"][1], size = (num_neurons,num_neurons) )
    syn_dict["E_syn"] = E_syn_inhib * (1 - weights_ranges_dict["types"]) + E_syn_excite * weights_ranges_dict["types"]
    syn_dict["g_syn_constant"] = log10uniform(syn_range_dict["g_syn_constant"][0], syn_range_dict["g_syn_constant"][1], size= (num_neurons,num_neurons) )

    weights_dict = dict()

    weights_dict["anatomy"] = weights_ranges_dict["anatomy"]
    weights_dict["types"] = weights_ranges_dict["types"]
    weights_dict["w"] = rectified_normal(weights_ranges_dict["w"][0], weights_ranges_dict["w"][1], (num_neurons, num_neurons))
    weights_dict["w_max"] = uniform(weights_ranges_dict["w_max"][0], weights_ranges_dict["w_max"][1], size = (num_neurons, num_neurons))
    weights_dict["tau_LTP"] = uniform(weights_ranges_dict["tau_LTP"][0], weights_ranges_dict["tau_LTP"][1], size = (num_neurons,))
    weights_dict["tau_LTD"] = uniform(weights_ranges_dict["tau_LTD"][0], weights_ranges_dict["tau_LTD"][1], size = (num_neurons,))
    weights_dict["tau_LTP_slow"] = uniform(weights_ranges_dict["tau_LTP_slow"][0], weights_ranges_dict["tau_LTP_slow"][1], size = (num_neurons,))
    weights_dict["A"] = log10uniform(weights_ranges_dict["A"][0], weights_ranges_dict["A"][1], size = (num_neurons, num_neurons))
    weights_dict["beta"] = log10uniform(weights_ranges_dict["beta"][0], weights_ranges_dict["beta"][1], size = (num_neurons, num_neurons))
    weights_dict["w_p"] = rectified_normal(weights_ranges_dict["w_p"][0], weights_ranges_dict["w_p"][1], size = (num_neurons, num_neurons))
    weights_dict["P"] = log10uniform(weights_ranges_dict["P"][0], weights_ranges_dict["P"][1], size = (num_neurons, num_neurons))
    weights_dict["tau_cons"] = uniform(weights_ranges_dict["tau_cons"][0], weights_ranges_dict["tau_cons"][1], size = (num_neurons, num_neurons))
    weights_dict["transmitter_constants"] = log10uniform(weights_ranges_dict["transmitter_constants"][0], weights_ranges_dict["transmitter_constants"][1], size = (num_neurons, num_neurons))
    weights_dict["tau_hom"] = uniform(weights_ranges_dict["tau_hom"][0], weights_ranges_dict["tau_hom"][1], size = (num_neurons, num_neurons))
    weights_dict["tau_ht"] = log10uniform(weights_ranges_dict["tau_ht"][0], weights_ranges_dict["tau_ht"][1], size = (num_neurons,))
    weights_dict["eta"] = log10uniform(weights_ranges_dict["eta"][0], weights_ranges_dict["eta"][1], size = (num_neurons, num_neurons))
    weights_dict["tau_H"] = uniform(weights_ranges_dict["tau_H"][0], weights_ranges_dict["tau_H"][1], size = (num_neurons, num_neurons))
    weights_dict["gamma"] = uniform(weights_ranges_dict["gamma"][0], weights_ranges_dict["gamma"][1], size = (num_neurons, num_neurons))

    configs = {
        "Ensemble": ensemble_dict,
        "SynapticCurrent": syn_dict,
        "Weights": weights_dict
    }
    return configs



def get_SMC_configs():
    num_neurons, ensemble_range, syn_ranges, weights_ranges = SMC_template()
    configs = gen_params_dict(num_neurons, ensemble_range, syn_ranges, weights_ranges)
    return num_neurons, configs

def convert_config2genes(configs):
    shapes_list = []
    keys_list = []
    separation_idx = [0]
    config_keys_list = []
    all_arrays_np = np.array([])

    for key_config in configs.keys():
        config = configs[key_config]

        for key in config.keys():
            shapes_list.append(config[key].shape)
            config_keys_list.append(key_config)
            keys_list.append(key)
            
            # get index
            arr_flatten = config[key].flatten()
            flatten_len = arr_flatten.shape[0]
            current_index = separation_idx[-1] + flatten_len
            separation_idx.append(current_index)

            all_arrays_np = np.append(all_arrays_np, arr_flatten)
    
    genes_dict = {
        "shapes": shapes_list,
        "config_keys": config_keys_list,
        "keys": keys_list,
        "separation_idx": separation_idx,
        "genes": all_arrays_np
    }
    return genes_dict

def convert_genes2config(genes_dict):
    unique_configs = set(genes_dict["config_keys"])
    configs = dict()
    for unique_config in unique_configs:
        configs[unique_config] = dict()
    length_list = len(genes_dict["shapes"])
    for idx in range(length_list):
        arr_start_idx, arr_end_idx = genes_dict["separation_idx"][idx], genes_dict["separation_idx"][idx+1]
        arr_retrieved = genes_dict["genes"][arr_start_idx: arr_end_idx].reshape(genes_dict["shapes"][idx])
        configs[genes_dict["config_keys"][idx]][genes_dict["keys"][idx]] = arr_retrieved
    
    return configs

def test_if_two_configs_are_equal(configs1, configs2):
    for config_key in configs1.keys():
        for params_key in configs1[config_key].keys():
            assert np.array_equal(configs1[config_key][params_key], configs2[config_key][params_key])
        
    

    

if __name__ == "__main__":
    num, configs = get_SMC_configs()
    genes_dict = convert_config2genes(configs)
    configs_recovered = convert_genes2config(genes_dict)