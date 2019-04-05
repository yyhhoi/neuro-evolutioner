import numpy as np
from .SynapticDynamics import FiringMask, WeightsDynamics


class Ensemble_AdEx(object):
    def __init__(self, simenv, num_neurons):
        self.num_neurons = num_neurons
        self.simenv = simenv

        # Firing
        self.firing_mask = FiringMask(num_neurons)
        self.firing_rate = np.zeros((num_neurons,))

        # External current
        self.I_ext = np.zeros((num_neurons))

        # Declare dynamics variables
        self.w = np.zeros(num_neurons)
        self.g = np.zeros((num_neurons, num_neurons))
        self.synaptic_currents = np.zeros((num_neurons))
        self.Weights = WeightsDynamics(self.num_neurons, self.simenv.epsilon)


    def state_update(self):
        self._threshold_crossing() # Register which neurons fire and reset potentials
        self.Weights.update(self.firing_mask)
        self._syn_current_dynamics_update() 
        self._membrane_potential_dyanmics_update()

        

    def _membrane_potential_dyanmics_update(self):

        non_linear_kernal = -(self.u - self.u_rest) + self.sharpness * np.exp((self.u-self.u_threshold)/self.sharpness)
        du_dt = (non_linear_kernal - self.r_m * self.w + self.r_m * self.I_ext - self.r_m * self.synaptic_currents)/self.tau_m
        
        dw_dt = (self.a * (self.u - self.u_rest) - self.w + (self.b * self.tau_w * self.firing_mask.get_mask()) )/self.tau_w
        
        self.u += du_dt * self.simenv.epsilon
        self.w += dw_dt * self.simenv.epsilon
        
    def _syn_current_dynamics_update(self):

        dg_dt = -self.g/self.tau_syn + (self.firing_mask.get_2d_cols() * self.g_syn_constant * self.Weights.get_weights())
        
        self.g += dg_dt * self.simenv.epsilon
        self._calc_synaptic_current()



    def _threshold_crossing(self):
        self.firing_mask.update_mask(self.u, self.u_threshold)
        self.u[self.firing_mask.get_mask()] = self.u_reset[self.firing_mask.get_mask()] 


    
    def _calc_synaptic_current(self):
        u_2d = np.repeat(self.u.reshape(1,-1), self.num_neurons, axis=0)  # turn self.u to (1, num_neurons), then expand
        current_2d = self.g * (u_2d - self.E_syn)
        current_1d = np.sum(current_2d, axis = 0)
        self.synaptic_currents = current_1d



    def initialize_parameters(self, configs):

        """
        Args:
            configs: (dict) Configures of parameters and constants. It contains all two dictionaries, linked by key 'Weights' and 'Ensemble'. 
                ['Ensemble']: All keys contain a (num_neurons, ) numpy array. Parameters for defining all neurons.
                    "u": ~ u_rest = -70mV = -70e-3
                    "u_rest": ~ -70mV = -70e-3
                    "r_m": 10Mohm - 1000Mohm = 10e6 - 1000e6
                    "tau_m": 5ms - 200ms = 5e-3 - 200e-3
                    "u_threshold": -50mV = -50e-3
                    "u_reset": -46mV - -60mV = -46e-3 - -60e-3
                    "sharpness": ~ 2mV = 2e-3
                    "tau_w": 30ms - 100ms = 30e-3 - 100e-3
                    "a": -1nS - 1nS = -1e-9 - 1e-9
                    "b": 5pA - 60pA = 5e-12 - 60e-12

                ['SynapticCurrent']: All keys contain a (num_neurons, num_neurons) numpy array.
                    "tau_syn": ~ 5mV = 5e-3
                    "E_syn": (Inhibitory) ~ -75mV = -75e-3 or (Excitatory) ~ 0mV = 0
                    "g_syn_constant": ~ 40pS = 40e-12, range = [4e-3, 4e-12]

                ['Weights']: All keys contain a (num_neurons, num_neurons) numpy array, 
                            except tau_LTP, tau_LTP_slow, tau_LTD, tau_ht are (num_neurons,) shape
                    "anatomy": Anatomical restriction. 0 = Never have any synapse. 1 = opposite
                    "types": 0 = inhibitory synapses, 1 = excitatory
                    "w": Initial weights
                    "w_max": maximum weights
                    "tau_LTP": 
                    "tau_LTP_slow": Must greater than tau_LTP
                    "tau_LTD":
                    "A": Excitatory combination 1. Internally copied to be parameter B (Excitatory combination 2)
                    "beta": Excitatory combination 3. Strength of heterosynaptic dynamics
                    "w_p": 0.5
                    "P": Consolidation strength
                    "tau_cons": 20mins = 1200
                    "transmitter_constants": Excitatory combination 4
                    "tau_hom": ~20 minutes = 1200
                    "tau_ht": ~100ms = 100e-3
                    "eta": Inhibition combination 1
                    "gamma":
                    "tau_H": ~10s = 10
        """

        self.configs = configs
        self._initialise_ensembles_params(
            u = configs["u_rest"],
            u_rest = configs["u_rest"],
            r_m = configs["r_m"],
            tau_m = configs["tau_m"],
            u_threshold = configs["u_threshold"],
            u_reset = configs["u_reset"],
            sharpness = configs["sharpness"],
            tau_w = configs["tau_w"],
            a = configs["a"],
            b = configs["b"]
            )
        self._initialise_synaptic_current_params(
            tau_syn = configs["tau_syn"],
            E_syn = configs["E_syn"],
            g_syn_constant = configs["g_syn_constant"]
            )
        self._initialise_weights(configs)
        

    def _initialise_ensembles_params(self, u, u_rest, r_m, tau_m, u_threshold, u_reset, sharpness, tau_w, a, b ):
        self.u, self.u_rest, self.r_m, self.tau_m, self.u_threshold, self.u_reset = u, u_rest, r_m, tau_m, u_threshold, u_reset
        self.sharpness, self.tau_w, self.a, self.b= sharpness, tau_w, a, b
    
    def _initialise_synaptic_current_params(self, tau_syn, E_syn, g_syn_constant):
        self.tau_syn, self.E_syn, self.g_syn_constant = tau_syn, E_syn, g_syn_constant
    
    def _initialise_weights(self, weights_config):
        self.Weights.set_configurations(weights_config)




    