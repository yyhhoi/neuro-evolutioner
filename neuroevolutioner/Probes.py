from .utils import write_pickle
    
    
class Probe():
    def __init__(self,
                 num_neurons,
                 activity_record_path,
                 stimuli_record_path,
                 gene_save_path
                 ):
        self.num_neurons = num_neurons
        self.activity_record_path = activity_record_path
        # self.firing_rate_path = firing_rate_path
        self.stimuli_record_path = stimuli_record_path
        self.gene_save_path = gene_save_path

        self._initialise_activity_handle()
        # self._initialise_firing_rate_handle()
        self._initialise_stimuli_handlle()



    def write_out_activity(self, time, condition, firing_masK_str):
        self.record_activity.write("{},{},".format(time, condition) + ",".join(firing_masK_str) + "\n")

    # def write_out_rate(self, time, firing_rate_str):
    #     self.record_rate.write("{},".format(time) + ",".join(firing_rate_str) + "\n")
    
    def write_out_stimuli(self, time, condition, label, I_ext_str):
        self.record_stimuli.write("{},{},{},".format(time, condition, label) + ",".join(I_ext_str) + "\n")

    def save_gene(self, gene):
        write_pickle(self.gene_save_path, gene)



    def _initialise_activity_handle(self):        
        num_list = ["time", "condition"] + ["neuron_{}".format(str(x+1)) for x in range(self.num_neurons)]
        self.record_activity = open(self.activity_record_path, "w")
        self.record_activity.write(",".join(num_list) + "\n")

    # def _initialise_firing_rate_handle(self):
    #     num_list = ["time"] + ["neuron_{}".format(str(x+1)) for x in range(self.num_neurons)]
    #     self.record_rate = open(self.firing_rate_path, "w")
    #     self.record_rate.write(",".join(num_list) + "\n")
    
    def _initialise_stimuli_handlle(self):
        num_list = ["time", "condition", "label"] + ["neuron_{}".format(str(x+1)) for x in range(self.num_neurons)]
        self.record_stimuli = open(self.stimuli_record_path, "w")
        self.record_stimuli.write(",".join(num_list) + "\n")

    def __del__(self):
        self.record_activity.close()
        # self.record_rate.close()
        self.record_stimuli.close()