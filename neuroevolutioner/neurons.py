import numpy as np
# Standard range of neuronal parameters can be referred to https://neuralensemble.org/docs/PyNN/0.7/standardmodels.html



class LIF_Neuron():
    def __init__(self, simenv, u_rest = -70, u_reset = -70, u_threshold = -50, tau_m = 10, r_m = 500):
        """
        Args:
            u_rest: Resting membrane potential. Default= -70ms
            u_threshold: Threshold of firing. Default= -50ms
            tau_m: Membrane time constant. Default= 20ms, typically 10 - 50ms
            r_m: Membrane resistance. Default value = 20M(Ohm). Further validation required, especially for the unit
        """
        # Pre-defined parameters
        self.u_rest = u_rest
        self.u_threshold = u_threshold
        self.tau_m = tau_m
        self.r_m = r_m
        self.u_reset = u_reset

        # Changing quantities
        self._u = self.u_rest
        self._I_ext = 0
        self.spike_occurred = 0

        # Simulation environment
        self.simenv = simenv
    def state_update(self):
        self._threshold_crossing()
        self._dynamics_update()

    def _dynamics_update(self):
        du_dt = -(self.u - self.u_rest)/self.tau_m + (self.r_m/self.tau_m) * self.I_ext
        self.u += (du_dt * self.simenv.epsilon)
        self.spike_occurred = 0

    def _threshold_crossing(self):
        if self.u > self.u_threshold:
            self.u = self.u_reset
            self.spike_occurred = 1
        
    @property
    def I_ext(self):
        return self._I_ext
    @I_ext.setter
    def I_ext(self, value):
        self._I_ext = value

    @property
    def u(self):
        return self._u
    @u.setter
    def u(self, value):
        self._u = value

class AdEx_Neuron(LIF_Neuron):
    def __init__(self, simenv, u_rest = -70, u_reset = -55, u_threshold = -50, tau_m = 200, r_m = 500,
                 sharpness = 2, tau_w = 100, a = 0, b = 5 ):
    
        # AdEx Neuron's default parameters for Adapting behaviour (Input current = 65pA)
        super(AdEx_Neuron, self).__init__(simenv, u_rest, u_reset, u_threshold, tau_m, r_m)
        self.sharpness = sharpness
        self.tau_w = tau_w
        self.a = a
        self.b = b

        # Dynamic quantities
        self.w = 0
        self.spike_occurred = 0

    def _dynamics_update(self):
        
        non_linear_kernal = -(self.u - self.u_rest) + self.sharpness * np.exp((self.u-self.u_threshold)/self.sharpness)

        du_dt = (non_linear_kernal - self.r_m * self.w + self.r_m * self.I_ext)/self.tau_m        
        dw_dt = (self.a * (self.u - self.u_rest) - self.w + (self.b * self.tau_w * self.spike_occurred) )/self.tau_w

        self.u += du_dt * self.simenv.epsilon
        self.w += dw_dt * self.simenv.epsilon

        self.spike_occurred = 0