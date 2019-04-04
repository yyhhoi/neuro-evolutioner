from neuroevolutioner.Environments import Simulation
from neuroevolutioner.Ensembles import Ensemble_AdEx
from neuroevolutioner.utils import load_pickle, write_pickle
from neuroevolutioner.ParamsGenomes import get_SG_configs, convert_config2genes, convert_genes2config
from neuroevolutioner.probes import Probe
from neuroevolutioner.params_templates.sequence_generation import SG_anatomy

from multiprocessing import Process

import argparse
import numpy as np
import os

class MapXY():
    def __init__(self):
        self.p0 = np.array([0,0])
        self.p1 = np.array([0,1])
        self.p2 = np.array([1,0])
        self.p3 = np.array([1,1])
        self.sq0 = np.zeros((3,2))
        self.sq1 = np.array([[0, 1], [1, 0], [1, 1]])
        self.sq2 = np.array([[1, 0], [1, 1], [0, 1]])
        self.sq3 = np.array([[1, 1], [0, 1], [1, 0]])

    def convert_single(self,x):
        if np.array_equal(x,self.p0):
            return self.sq0
        elif np.array_equal(x,self.p1):
            return self.sq1
        elif np.array_equal(x,self.p2):
            return self.sq2
        elif np.array_equal(x,self.p3):
            return self.sq3


def load_data(train_m = 100, test_m = 100):
    X_train = np.random.randint(0,2, (train_m, 2))
    Y_train = np.zeros((train_m, 3, 2))
    X_test = np.random.randint(0,2, (test_m, 2))
    Y_test = np.zeros((test_m, 3, 2))
    mapxy = MapXY()
    for i in range(train_m):
        X_train_each = X_train[i, ]
        Y_train[i, ] = mapxy.convert_single(X_train_each)
    for i in range(test_m):
        X_test_each = X_test[i, ]
        Y_test[i, ] = mapxy.convert_single(X_test_each)
    return (X_train, Y_train, train_m), (X_test, Y_test, test_m)


def gen_condition_time_list():
    def appending(key_to_append):
        times_list.append(times_list[-1] + conditions[key_to_append])
        conditions_list.append(key_to_append)
    conditions = {
        "train_sti": 1, #
        "train_response1": 0.5,
        "train_response2": 0.5,
        "train_response3": 0.5,
        "train_ISI": 0.5,
        "train_sti_repeat": 40,
        "rest": 5,
        "test_sti": 1,
        "test_response1": 0.5,
        "test_response2": 0.5,
        "test_response3": 0.5,
        "test_ISI": 0.5,
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

        appending("train_response1")
        appending("train_response2")
        appending("train_response3")
        appending("train_ISI")
    # Rest
    appending("rest")

    # Testing stimulus and ISI repeated for 20 times
    for i in range(conditions["test_sti_repeat"]):
        appending("test_sti")
        appending("test_response1")
        appending("test_response2")
        appending("test_response3")
        appending("test_ISI")
    return times_list, conditions_list, conditions



class Experimenter():
    def __init__(self, num_neurons, anatomy_labels):
        (self.X_train, self.Y_train, self.train_m), (self.X_test, self.Y_test, self.test_m ) = load_data()

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
            # print("%s: %0.5f < %0.5f < %0.5f" % ("Condition", self.times_list[self.current_time_idx-1], time, self.times_list[self.current_time_idx]) )
            self.condition = self.conditions_list[self.current_time_idx]    
            if self.condition == "train_sti":
                
                stimuli_patten = self.X_train[self.sampled_idxes_train[self.current_sampled_idx_train], ]
                action_sequence = self.Y_train[self.sampled_idxes_train[self.current_sampled_idx_train], ]
                self.label =  "train_stimulus_" + "".join(stimuli_patten.flatten().astype(int).astype(str))
                self._update_I_ext_stimuli(stimuli_patten)

            elif self.condition == "train_response1":
                action_sequence = self.Y_train[self.sampled_idxes_train[self.current_sampled_idx_train], ]
                self.label = "train_response_" + "".join(action_sequence[0, ].flatten().astype(int).astype(str))
                self._update_I_ext_response(action_sequence[0, ])

            elif self.condition == "train_response2":
                action_sequence = self.Y_train[self.sampled_idxes_train[self.current_sampled_idx_train], ]
                self.label = "train_response_" + "".join(action_sequence[1, ].flatten().astype(int).astype(str))
                self._update_I_ext_response(action_sequence[1, ])

            elif self.condition == "train_response3":
                action_sequence = self.Y_train[self.sampled_idxes_train[self.current_sampled_idx_train], ]
                self.label = "train_response_" + "".join(action_sequence[2, ].flatten().astype(int).astype(str))
                self._update_I_ext_response(action_sequence[2, ])
                self.current_sampled_idx_train += 1

            elif self.condition == "test_sti":
                stimuli_patten = self.X_test[self.sampled_idxes_test[self.current_sampled_idx_test], ]
                self.label =  "test_stimulus_" + "".join(stimuli_patten.flatten().astype(int).astype(str))
                self._update_I_ext_stimuli(stimuli_patten)

            elif self.condition == "test_response1":
                action_sequence = self.Y_test[self.sampled_idxes_test[self.current_sampled_idx_test], ]
                self.label = "test_response_" + "".join(action_sequence[0, ].flatten().astype(int).astype(str))
                self.I_ext = np.zeros(self.num_neurons)
            elif self.condition == "test_response2":
                action_sequence = self.Y_test[self.sampled_idxes_test[self.current_sampled_idx_test], ]
                self.label = "test_response_" + "".join(action_sequence[1, ].flatten().astype(int).astype(str))
                self.I_ext = np.zeros(self.num_neurons)
            elif self.condition == "test_response3":
                action_sequence = self.Y_test[self.sampled_idxes_test[self.current_sampled_idx_test], ]
                self.label = "test_response_" + "".join(action_sequence[2, ].flatten().astype(int).astype(str)) 
                self.I_ext = np.zeros(self.num_neurons)   
                self.current_sampled_idx_test += 1
            else:
                self.label = np.nan
                self.I_ext = np.zeros(self.num_neurons)
            
            self.current_time_idx += 1 # This is important - to ensure efficient increment

        return time, self.condition, self.label,  self.I_ext

    def _update_I_ext_stimuli(self, stimuli_patten): # stimuli_pattern = ndarray(0, 1)... etc
        stimulus = np.zeros(self.num_neurons)
        
        stimulus[0:self.anatomy_labels["sensory1"]] = stimuli_patten[0]
        stimulus[self.anatomy_labels["sensory1"]:self.anatomy_labels["sensory2"]] = stimuli_patten[1]
        self.I_ext = stimulus
    def _update_I_ext_response(self, action_sequence_each): # e.g. action_sequence_each = ndarray(1,1)
        stimulus = np.zeros(self.num_neurons)
        action_sequence_each[action_sequence_each==0] = -1
        stimulus[self.anatomy_labels["brain"]:self.anatomy_labels["class1"]] = action_sequence_each[0]
        stimulus[self.anatomy_labels["class1"]:self.anatomy_labels["class2"]] = action_sequence_each[1]
        self.I_ext = stimulus


def simulate_and_get_activity(data_dir, gen_idx = 1, species_idx = 1, time_step = 0.0001):
    
    # Define storage directories
    os.makedirs(data_dir, exist_ok=True)
    activity_record_path = os.path.join(data_dir, "activity.csv")
    firing_rate_path = os.path.join(data_dir, "firing_rate.csv")
    stimuli_path = os.path.join(data_dir, "stimuli.csv")
    gene_save_path = os.path.join(data_dir, "gene.pickle")
    
    # Sample from gen template distributions to create configuration of the species
    configs = get_SG_configs()
    num_neurons = configs["Meta"]["num_neurons"]
    anatomy_matrix, anatomy_labels = SG_anatomy()

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
        ensemble.I_ext = I_ext * 1
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


def proliferate_one_generation(project_results_dir, gen_idx=1, num_species=100, time_step=0.0005):


    for species_idx in range(num_species):
        data_dir = os.path.join(project_results_dir, "generation_{}/species_{}/".format(gen_idx, species_idx+1))
        os.makedirs(data_dir, exist_ok=True)
        if os.path.isfile(os.path.join(data_dir, "gene.pickle")) :
            print("Generation {} and species {} exist. Skipped".format(gen_idx, species_idx))
        else:
            print("Generation: {} | Species: {}/{}".format(gen_idx, species_idx, num_species))
            simulate_and_get_activity(data_dir,  gen_idx=gen_idx, species_idx=species_idx, time_step=time_step)

if __name__ == "__main__":

    project_results_dir = "experiment_results/sequence_generation"
    proliferate_one_generation(project_results_dir)

    
    # # Debug of times and conditions list
    # times_list, conditions_list, conditions = gen_condition_time_list()
    # for time,cond in zip(times_list, conditions_list):
    #     print("time: {} | cond: {}".format(time, cond))

    # # Debug of experimenter
    # configs = get_SG_configs()
    # num_neurons = configs["Meta"]["num_neurons"]
    # anatomy_matrix, anatomy_labels = SG_anatomy()
    # exper = Experimenter(num_neurons, anatomy_labels)
    # time_step = 0.1
    # simenv = Simulation(exper.max_time, epsilon=time_step)
    # while simenv.sim_stop == False:
    #     time  = simenv.getTime()
    #     # Get current conditions and amount of external currents, given current time
    #     _, condition, label,  I_ext = exper.get_stimulation_info(time)
    #     print("time: %0.3f | Condition: %s | label: %s\n I_ext_sen:\n%s\n I_ext_act:\n%s" %(time, condition, label,
    #                                                                                         str(list(I_ext[0:anatomy_labels["sensory2"]])),
    #                                                                                         str(list(I_ext[anatomy_labels["brain"]:anatomy_labels["class2"]]))
    #                                                                                         ))
    #     simenv.increment()