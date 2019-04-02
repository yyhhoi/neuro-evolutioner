from neuroevolutioner.Environments import Simulation
from neuroevolutioner.Ensembles import Ensemble_AdEx
from neuroevolutioner.utils import load_pickle, write_pickle
from neuroevolutioner.ParamsGenomes import get_SMC_configs, convert_config2genes, convert_genes2config, test_if_two_configs_are_equal
from neuroevolutioner.probes import Probe
from neuroevolutioner.params_templates.simple_minist_classification import SMC_anatomy

import numpy as np
from sklearn import datasets
import os

def load_data(split_percent, n_class=4):
    digits = datasets.load_digits(n_class= n_class)
    X = digits.images/16 # Normalise to [0,1]
    X = X.reshape(X.shape[0], 64)
    Y = digits.target
    
    split_idx = int(split_percent*X.shape[0])

    X_train, Y_train = X[:split_idx, ], Y[:split_idx]
    X_test, Y_test = X[split_idx:, ], Y[split_idx: ]

    return (X_train, Y_train, X_train.shape[0]), (X_test, Y_test, X_test.shape[0])

def experiment_details():
    pass

def gen_condition_time_list():
    conditions = {
        "train_sti": 1,
        "train_ISI": 0.5,
        "train_sti_repeat": 80,
        "rest": 5,
        "test_sti": 0.5,
        "test_ISI": 0.1,
        "test_sti_repeat": 20
    }

    times_list = []
    conditions_list = []
    # Training stimulus and ISI repeated for many times
    for i in range(conditions["train_sti_repeat"]):
        if i != 0:
            times_list.append(times_list[-1] + conditions["train_sti"])
            
        else:
            times_list.append(0 + conditions["train_sti"])
            
        conditions_list.append("train_sti")
        times_list.append(times_list[-1] + conditions["train_ISI"])
        conditions_list.append("train_ISI")
    
    # Rest
    times_list.append(times_list[-1] + conditions["rest"])
    conditions_list.append("rest")

    # Testing stimulus and ISI repeated for 20 times
    for i in range(conditions["test_sti_repeat"]):
        times_list.append(times_list[-1] + conditions["test_sti"])
        conditions_list.append("test_sti")
        times_list.append(times_list[-1] + conditions["test_ISI"])
        conditions_list.append("test_ISI")
    return times_list, conditions_list, conditions


class Experimenter():
    def __init__(self, num_neurons, anatomy_labels, n_class= 4):
        (self.X_train, self.Y_train, self.train_m), (self.X_test, self.Y_test, self.test_m ) = load_data(split_percent=0.8, n_class=4)

        self.num_neurons = num_neurons
        self.anatomy_labels = anatomy_labels
        self.times_list, self.conditions_list, self.conditions_dict = gen_condition_time_list()
        self.I_ext = np.zeros(num_neurons)
        self.sampled_idxes_train = np.random.choice(self.X_train.shape[0], self.conditions_dict["train_sti_repeat"], replace=False)
        self.sampled_idxes_test = np.random.choice(self.X_test.shape[0], self.conditions_dict["test_sti_repeat"], replace=False)

        self.max_time_idx = len(self.times_list)
        self.max_time = max(self.times_list)
        self.current_sampled_idx_train = 0
        self.current_sampled_idx_test = 0
        self.current_time_idx = 0
        self.label = np.nan
        self.condition = "rest"
    def get_stimulation_info(self, time):
        
        
        if self.current_time_idx == 0:
            condition_statement = ((time > 0) and (time < self.times_list[self.current_time_idx]))

        elif self.current_time_idx < self.max_time_idx:
            condition_statement = ((time > self.times_list[self.current_time_idx-1]) and (time < self.times_list[self.current_time_idx]))

        else:
            condition_statement = False

        if condition_statement:
            # print("{} < {}".format(time, self.times_list[self.current_time_idx]))
            self.condition = self.conditions_list[self.current_time_idx]    
            if self.condition == "train_sti":
                self.label = self.Y_train[self.sampled_idxes_train[self.current_sampled_idx_train]]
                self._update_I_ext(self.X_train[self.sampled_idxes_train[self.current_sampled_idx_train], ], self.label)
                
                self.current_sampled_idx_train += 1
                
            elif self.condition == "test_sti":
                self.label = self.Y_test[self.sampled_idxes_test[self.current_sampled_idx_test]]
                self._update_I_ext(self.X_test[self.sampled_idxes_test[self.current_sampled_idx_test], ], self.label)
                
                self.current_sampled_idx_test += 1
            else:
                self.label = np.nan
                self.I_ext = np.zeros(self.num_neurons)
            
            self.current_time_idx += 1

        return time, self.condition, self.label,  self.I_ext
    def _update_I_ext(self, stimulus, action_stimuli_idxes):
        start, end = self._convert_class_to_index(action_stimuli_idxes)
        self.I_ext[0:64] = stimulus 
        if start is not None:
            self.I_ext[start: end] = 1


    def _convert_class_to_index(self, digit_class):
        
        if digit_class == 0:
            start, end = self.anatomy_labels["brain"], self.anatomy_labels["class1"]
        elif digit_class == 1:
            start, end = self.anatomy_labels["class1"], self.anatomy_labels["class2"]
        elif digit_class == 2:
            start, end = self.anatomy_labels["class2"], self.anatomy_labels["class3"]
        elif digit_class == 3:
            start, end = self.anatomy_labels["class3"], self.anatomy_labels["class4"]
        else:
            start, end = None, None
        
        return start, end

def simulate_and_get_activity(num_gen = 1, num_species = 100, time_step = 0.00001):
        
        for gen_idx in range(num_gen):
            for species_idx in range(num_species):
                # Print out curent session
                print("Generation: {}/{} | Species: {}/{}".format(gen_idx+1, num_gen, species_idx+1, num_species))

                # Define storage directories
                data_dir = "experiment_results/simple_minist_classification/generation_{}/species_{}/".format(gen_idx+1, species_idx+1)
                os.makedirs(data_dir, exist_ok=True)
                activity_record_path = os.path.join(data_dir, "activity.csv")
                firing_rate_path = os.path.join(data_dir, "firing_rate.csv")
                stimuli_path = os.path.join(data_dir, "stimuli.csv")
                gene_save_path = os.path.join(data_dir, "gene.pickle")
                
                # Sample from gen template distributions to create configuration of the species
                configs = get_SMC_configs()
                num_neurons = configs["Meta"]["num_neurons"]
                anatomy_matrix, anatomy_labels = SMC_anatomy()

                # Initialise experimental paradigm
                exper = Experimenter(num_neurons, anatomy_labels)

                # Initialise simulation environment and neuronal ensemble
                simenv = Simulation(exper.max_time, epsilon=time_step)
                ensemble = Ensemble_AdEx(simenv, num_neurons, configs)
                ensemble.initialize_parameters(configs)
                
                # Initialise probing
                probe = Probe(
                    num_neurons = num_neurons,
                    activity_record_path = activity_record_path,
                    stimuli_record_path = stimuli_path,
                    gene_save_path = gene_save_path
                )
                
                # Simulation starts
                while simenv.sim_stop == False:
                    time  = simenv.getTime()
                    # Print progress
                    print("\r{}/{}".format(time,exper.max_time), flush=True, end="")
                    
                    # Get current conditions and amount of external currents, given current time
                    _, condition, label,  I_ext = exper.get_stimulation_info(time)

                    # Apply current and update the dynamics
                    ensemble.I_ext = I_ext * 1e-9
                    ensemble.state_update()

                    # Increment simulation environment
                    simenv.increment()

                    # Write out records
                    probe.write_out_activity(time, ensemble.firing_mask.get_mask().astype(int).astype(str))
                    probe.write_out_stimuli(time, condition, label, ensemble.I_ext.astype(str))
                    
                
                # Save genes
                print("\nSaving the genes")
                gene = convert_config2genes(configs)
                probe.save_gene(gene)




if __name__ == "__main__":
    
    simulate_and_get_activity()