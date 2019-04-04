import numpy as np

class FiringMask():
    def __init__(self, num_neurons):
        self.firing_mask_2d_template = np.zeros((num_neurons, num_neurons))
        self.firing_mask = np.zeros(num_neurons)
    def update_mask(self, u, u_threshold):
        self.firing_mask =  u >  u_threshold # firing mask ~ (num_fired, )
    
    def get_mask(self):
        return self.firing_mask
    def get_2d_rows(self): # Expanded to multiple rows
        firing_mask_2d = self.firing_mask_2d_template.copy()
        firing_mask_2d[:, self.firing_mask == 1] = 1
        return firing_mask_2d

    def get_2d_cols(self):
        firing_mask_2d = self.firing_mask_2d_template.copy()
        
        firing_mask_2d[self.firing_mask == 1, :] = 1
        return firing_mask_2d

class SpikeTrace():

    def __init__(self, num_neurons, tau, time_step):
        """
        Args:
            num_neurons: (int) Number of Neurons
            tau: (ndarray) with shape (num_neurons,). Time constants for the each neuron's traces dynamics
            time_step: (float) Time step of the simulation environment
        """
        self.z = np.zeros(num_neurons) # Initialise traces 
        self.z_previous = self.z.copy()
        self.tau = tau
        self.time_step = time_step
    def increment(self, firing_mask):
        """
        Args:
            firing_mask: (ndarray) with shape (num_neurons,)
        """
        self.z_previous = self.z.copy()
        self.dz_dt = -self.z/self.tau + firing_mask.astype(int)
        self.z += self.dz_dt * self.time_step

        # Expansion
        self.z_2d_rows = np.repeat(self.z.reshape(1,-1), self.z.shape[0], axis=0)
        self.z_2d_cols = np.repeat(self.z.reshape(-1, 1), self.z.shape[0], axis=1)
        self.z_2d_rows_previous = np.repeat(self.z_previous.reshape(1,-1), self.z_previous.shape[0], axis=0)
        self.z_2d_cols_previous = np.repeat(self.z_previous.reshape(-1, 1), self.z_previous.shape[0], axis=1)

    def get_trace(self):
        return self.z
        
    def get_2d_rows(self):
        return self.z_2d_rows
    def get_2d_cols(self):
        return self.z_2d_cols
    
    def get_trace_previous(self):
        return self.z_previous
        
    def get_2d_rows_previous(self):
        return self.z_2d_rows_previous
    def get_2d_cols_previous(self):
        return self.z_2d_cols_previous



class WeightsDynamics():
    def __init__(self, num_neurons, time_step):
        self.num_neurons = num_neurons
        self.w = np.zeros((self.num_neurons, self.num_neurons))
        self.time_step = time_step

        # Triplet LTP and Douplet LTD
        self.B = np.zeros((self.num_neurons, self.num_neurons))

        # Heterosynaptic platiscity
        self.w_snake = np.zeros((self.num_neurons, self.num_neurons))

        # Homeostatic synapse
        self.C = np.zeros((self.num_neurons, self.num_neurons))
        
        # Long-term inhibition
        self.H = np.zeros((self.num_neurons, self.num_neurons))


    def update(self, firing_mask_instance):
        firing_mask_1d, firing_mask_2d_rows, firing_mask_2d_cols = firing_mask_instance.get_mask(), firing_mask_instance.get_2d_rows(), firing_mask_instance.get_2d_cols()
        
        self.z_LTP.increment(firing_mask_1d)
        self.z_LTP_slow.increment(firing_mask_1d)
        self.z_LTD.increment(firing_mask_1d)
        self.z_ht.increment(firing_mask_1d)
        self._update_w_snake()
        self._update_homeo()
        self._update_inhibition(firing_mask_2d_rows)

        # Excitatory synaptic dynamics
        dw_dt_LTP = self.A * self.z_LTP.get_2d_cols() * self.z_LTP_slow.get_2d_rows_previous() * firing_mask_2d_rows
        dw_dt_LTD = -self.B * self.z_LTD.get_2d_rows() * firing_mask_2d_cols
        dw_dt_hetero = -self.beta * (self.w - self.w_snake) * np.power(self.z_LTD.get_2d_rows_previous(), 3) * firing_mask_2d_rows
        dw_dt_transmitter = self.transmitter_constants * firing_mask_2d_cols
        dw_dt_excitatory = self.types * (dw_dt_LTP + dw_dt_LTD + dw_dt_hetero + dw_dt_transmitter)

        # Inhibitory synaptic dynamics
        trace_term = (self.z_LTP.get_2d_rows() + 1) * firing_mask_2d_cols + self.z_LTP.get_2d_cols() * firing_mask_2d_rows
        dynamics_term = self.eta * (self.H - self.gamma)
        dw_dt_inhibitory = (1 - self.types) * dynamics_term * trace_term

        # Update weights with anatomical constriants and hard bound
        self.w += (dw_dt_excitatory + dw_dt_inhibitory) * self.time_step
        self.w = self.anatomy * np.clip(self.w, 0, self.w_max)

    def set_configurations(self, configs):
        """
        Args:
            configs: (dict) Contains all configures of parameters and constants. 
            All keys contain a (num_neurons, num_neurons) numpy array.
                "anatomy": Anatomical restriction. 0 = Never have any synapse. 1 = opposite
                "types": 0 = inhibitory synapses, 1 = excitatory
                "w": Initial weights
                "w_max": maximum weights
                "tau_LTP": 
                "tau_LTP_slow": Must greater than tau_LTP
                "tau_LTD":
                "A": Excitatory combination 1. Internally copied to be parameter B (Excitatory combination 2)
                "beta":
                "w_p":
                "P": Excitatory combination 3. Strength of heterosynaptic dynamics
                "tau_cons":
                "transmitter_constants": Excitatory combination 4
                "tau_hom": ~20 minutes = 1200
                "tau_ht": ~100ms = 100e-3
                "eta": Inhibition combination 1
                "gamma":
                "tau_H": ~10s = 10

        """
        self._initialise_params(
            anatomy = configs["anatomy_matrix"], types = configs["types_matrix"], w = configs["w"], w_max = configs["w_max"],
            tau_LTP = configs["tau_LTP"], tau_LTP_slow = configs["tau_LTP_slow"], tau_LTD = configs["tau_LTD"], A = configs["A"],
            beta = configs["beta"], w_p = configs["w_p"], P = configs["P"], tau_cons = configs["tau_cons"],
            transmitter_constants = configs["transmitter_constants"],
            tau_hom = configs["tau_hom"], tau_ht = configs["tau_ht"],
            eta = configs["eta"], gamma = configs["gamma"], tau_H = configs["tau_H"]
            )
        pass

    def _initialise_params(self,
                          anatomy, types, w, w_max,
                          tau_LTP, tau_LTP_slow, tau_LTD, A,
                          beta, w_p, P, tau_cons,
                          transmitter_constants,
                          tau_hom, tau_ht,
                          eta, gamma, tau_H
                          ):
        """
        Args:
            A, B, beta, transmitter_constants, eta: recommended > 0
        """
        self.w, self.w_max, self.anatomy, self.types = w, w_max, anatomy, types

        # LTP/LTD
        self.z_LTP = SpikeTrace(self.num_neurons, tau_LTP, self.time_step)
        self.z_LTP_slow = SpikeTrace(self.num_neurons, tau_LTP_slow, self.time_step)
        self.z_LTD = SpikeTrace(self.num_neurons, tau_LTD, self.time_step)
        self.A, self.B  = A, A

        # Hetero
        self.beta, self.w_p, self.P, self.tau_cons = beta, w_p, P, tau_cons

        # Transmitters
        self.transmitter_constants = transmitter_constants
        
        # Homeostatic
        self.z_ht = SpikeTrace(self.num_neurons, tau_ht, self.time_step)
        self.tau_hom = tau_hom
        
        # Long-term inhibition
        self.eta, self.gamma, self.tau_H = eta, gamma, tau_H 
        
    
    def _update_w_snake(self):
        dwsnake_dt = self.w - self.w_snake - self.P * self.w_snake * (self.w_p/2 - self.w_snake) * (self.w_p - self.w_snake)
        self.w_snake += (dwsnake_dt / self.tau_cons) * self.time_step
            
    def _update_homeo(self):
        dC_dt = -self.C/self.tau_hom + np.square(self.z_ht.get_2d_rows())
        self.C += dC_dt * self.time_step
        self.B = self.A.copy()
        C_mask = self.C<1
        self.B[C_mask] = self.B[C_mask] * self.C[C_mask]
    
    def _update_inhibition(self, firing_mask_2d_rows):

        dH_dt = -self.H/self.tau_H + self.types * firing_mask_2d_rows
        self.H += dH_dt * self.time_step
    
    def get_weights(self):
        return self.w
