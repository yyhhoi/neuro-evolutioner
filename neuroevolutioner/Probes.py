from .utils import write_pickle
import numpy as np
import os
class Probe():
    def __init__(self,
                 num_neurons,
                 activity_record_path = None,
                 stimuli_record_path = None,
                 weights_record_dir = None,
                 gene_save_path = None
                 ):
        self.num_neurons = num_neurons
        
        self.activity_record_path = activity_record_path
        self.stimuli_record_path = stimuli_record_path
        self.weights_record_dir = weights_record_dir
        self.gene_save_path = gene_save_path


        if activity_record_path:
            self._initialise_activity_handle()
        if stimuli_record_path:
            self._initialise_stimuli_handlle()



    def write_out_activity(self, time, condition, firing_masK_str):
        self.record_activity.write("{},{},".format(time, condition) + ",".join(firing_masK_str) + "\n")

    def write_out_stimuli(self, time, condition, label, I_ext_str):
        self.record_stimuli.write("{},{},{},".format(time, condition, label) + ",".join(I_ext_str) + "\n")

    def save_gene(self, gene):
        write_pickle(self.gene_save_path, gene)
    def save_weights(self, filename, weights):
        save_path = os.path.join(self.weights_record_dir, filename)
        np.save(save_path, weights)


    def _initialise_activity_handle(self):        
        num_list = ["time", "condition"] + ["neuron_{}".format(str(x+1)) for x in range(self.num_neurons)]
        self.record_activity = open(self.activity_record_path, "w")
        self.record_activity.write(",".join(num_list) + "\n")

    
    def _initialise_stimuli_handlle(self):
        num_list = ["time", "condition", "label"] + ["neuron_{}".format(str(x+1)) for x in range(self.num_neurons)]
        self.record_stimuli = open(self.stimuli_record_path, "w")
        self.record_stimuli.write(",".join(num_list) + "\n")

    def __del__(self):
        if self.activity_record_path:
            self.record_activity.close()

        if self.stimuli_record_path:
            self.record_stimuli.close()