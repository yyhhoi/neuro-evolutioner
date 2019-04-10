from abc import ABC, abstractmethod
from .experiment_configs.TL import TL_conditions_dict, TL_conditions_order
from .experiment_configs.DA import DA_conditions_dict, DA_conditions_order
import numpy as np

class Experimenter(ABC):
    def __init__(self, num_neurons, anatomy_labels):
        self.num_neurons = num_neurons
        self.anatomy_labels = anatomy_labels
        self.times_list, self.conditions_list, self.conditions_dict = self._gen_condition_time_list()
        self.max_time_idx = len(self.times_list)
        self.max_time = max(self.times_list)
        self.current_time_idx = 0
        self.label = np.nan
        self.condition = "rest"
        self.I_ext = np.zeros(num_neurons)
    def get_stimulation_info(self, time):
               
        if self.current_time_idx == 0:
            condition_statement = ((time > 0) and (time < self.times_list[self.current_time_idx]))

        elif self.current_time_idx < self.max_time_idx:
            condition_statement = ((time > self.times_list[self.current_time_idx-1]) and (time < self.times_list[self.current_time_idx]))
        else:
            condition_statement = False

        if condition_statement:
            self.condition = self.conditions_list[self.current_time_idx]    
            self._update_I_ext_stimuli(self.condition)
            self._update_label(self.condition)
            self.current_time_idx += 1 # This is important - to ensure efficient increment
        return time, self.condition, self.label,  self.I_ext
    
    @abstractmethod
    def _update_I_ext_stimuli(self, condition): 
        # Define your I_ext according to the condition
        self.I_ext = np.zeros(self.num_neurons)
    
    @abstractmethod   
    def _update_label(self, condition):
        # Define your label according to condition. If condition is self-explanatory, then this section may not be necessary
        self.label = np.nan
    
    @abstractmethod
    def _gen_condition_time_list(self):
        times_list = []
        conditions_list = []
        conditions = dict()
        return times_list, conditions_list, conditions
    
    @staticmethod
    def _appending(times_list, conditions_list, conditions_dict, key_to_append, start_time):
        times_list.append(start_time + conditions_dict[key_to_append])
        conditions_list.append(key_to_append)
        return times_list, conditions_list


class TL_Experimenter(Experimenter):
    def __init__(self, num_neurons, anatomy_labels):
        super(TL_Experimenter, self).__init__(num_neurons, anatomy_labels)
    
    def _update_I_ext_stimuli(self, condition):
        stimulus = np.zeros(self.num_neurons)
        if (condition == "train_S") or (condition == "test_S"):
            stimulus[0:self.anatomy_labels["sensory1"]] = 1
        elif condition == "train_A":
            stimulus[self.anatomy_labels["brain2"]: self.anatomy_labels["action1"]] = 1
        self.I_ext = stimulus
    def _update_label(self, condition):
        self.label = np.nan
    def _gen_condition_time_list(self):
        times_list, conditions_list = [], []
        conditions_dict = TL_conditions_dict
        for idx, condition_key in enumerate(TL_conditions_order):
            if idx == 0:
                self._appending(times_list, conditions_list, TL_conditions_dict, condition_key, 0)
            else:
                self._appending(times_list, conditions_list, TL_conditions_dict, condition_key, times_list[-1])
        return times_list, conditions_list, conditions_dict

class DA_Experimenter(Experimenter):
    def __init__(self, num_neurons, anatomy_labels):
        super(DA_Experimenter, self).__init__(num_neurons, anatomy_labels)
    
    def _update_I_ext_stimuli(self, condition):
        stimulus = np.zeros(self.num_neurons)
        if (condition == "S"):
            stimulus[0:self.anatomy_labels["sensory1"]] = 1
        self.I_ext = stimulus
    def _update_label(self, condition):
        self.label = np.nan
    def _gen_condition_time_list(self):
        times_list, conditions_list = [], []
        conditions_dict = DA_conditions_dict
        for idx, condition_key in enumerate(DA_conditions_order):
            if idx == 0:
                self._appending(times_list, conditions_list, DA_conditions_dict, condition_key, 0)
            else:
                self._appending(times_list, conditions_list, DA_conditions_dict, condition_key, times_list[-1])
        return times_list, conditions_list, conditions_dict