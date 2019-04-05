from neuroevolutioner.Environments import Simulation
from neuroevolutioner.Ensembles import Ensemble_AdEx
from neuroevolutioner.utils import load_pickle, write_pickle
from neuroevolutioner.genetics import initialise_TL_params
from neuroevolutioner.probes import Probe

import argparse
import numpy as np
import os


def gen_condition_time_list():
    def appending(key_to_append):
        times_list.append(times_list[-1] + conditions[key_to_append])
        conditions_list.append(key_to_append)
    conditions = {
        "train_S": 1,
        "train_ISI": 0.5,
        "train_A": 0.5,
        "rest1": 2,
        "test_S": 1,
        "test_ISI": 0.5,
        "Test_A":0.5,
        "repeat": 1
    }

    times_list = []
    conditions_list = []
    # Training stimulus and ISI repeated for many times
    for i in range(conditions["repeat"]):
        times_list.append(0 + conditions["train_S"])
        conditions_list.append("train_S")
        appending("train_ISI")
        appending("train_A")
        appending("rest1")
        appending("test_S")
        appending("test_ISI")
        appending("Test_A")
    return times_list, conditions_list, conditions



class Experimenter():
    def __init__(self, num_neurons, anatomy_labels):

        self.num_neurons = num_neurons
        self.anatomy_labels = anatomy_labels
        self.times_list, self.conditions_list, self.conditions_dict = gen_condition_time_list()
        self.I_ext = np.zeros(num_neurons)

        self.max_time_idx = len(self.times_list)
        self.max_time = max(self.times_list)
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
            self._update_I_ext_stimuli(self.condition)
            
            self.current_time_idx += 1 # This is important - to ensure efficient increment

        return time, self.condition, self.label,  self.I_ext

    def _update_I_ext_stimuli(self, condition): # stimuli_pattern = ndarray(0, 1)... etc
        stimulus = np.zeros(self.num_neurons)
        if (condition == "train_S") or (condition == "test_S"):
            stimulus[0:self.anatomy_labels["sensory1"]] = 1
        elif condition == "train_A":
            stimulus[self.anatomy_labels["brain2"]: self.anatomy_labels["action1"]] = 1

        self.I_ext = stimulus



def simulate_and_get_activity(data_dir, gen_idx = 1, species_idx = 1, time_step = 0.0005):
    
    # Define storage directories
    os.makedirs(data_dir, exist_ok=True)
    activity_record_path = os.path.join(data_dir, "activity.csv")
    firing_rate_path = os.path.join(data_dir, "firing_rate.csv")
    stimuli_path = os.path.join(data_dir, "stimuli.csv")
    gene_save_path = os.path.join(data_dir, "gene.pickle")
    
    # Sample from gen template distributions to create configuration of the species
    configs = initialise_TL_params()
    num_neurons = configs["num_neurons"]
    anatomy_matrix, anatomy_labels = configs["anatomy_matrix"], configs["anatomy_labels"]

    # Initialise experimental paradigm
    exper = Experimenter(num_neurons, anatomy_labels)

    # Initialise simulation environment and neuronal ensemble
    simenv = Simulation(exper.max_time, epsilon=time_step)
    ensemble = Ensemble_AdEx(simenv, num_neurons)
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
        probe.write_out_activity(time, condition, ensemble.firing_mask.get_mask().astype(int).astype(str))
        
    
    # Save genes
    print("\nSaving the genes")
    probe.save_gene(configs)


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

    project_results_dir = "experiment_results/time_learning"
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