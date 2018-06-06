import numpy as np
import simpy
import matplotlib.pyplot as plt
import pdb
from vec_params import *

# we denote vector variables with capital letters, while scalar with small letters
# Currently: tau_m, 
class Ensemble(object):
    def __init__(self, env, u_ini, num_neurons = test_num_neurons, u_rest = u_rest, leak_R_membrane = leak_R_membrane,
                tau_membrane = tau_membrane, u_threshold = u_threshold, u_reset = u_reset, time_step = epsilon):
        self.env = env
        self.U = np.ones((num_neurons,1)) * u_ini
        self.U_REST = np.ones((num_neurons, 1)) * u_rest
        self.M_R = np.ones((num_neurons, 1)) * leak_R_membrane
        self.tau_membrane, self.u_threshold, self.u_reset = tau_membrane, u_threshold, u_reset
        self.CURRENT_EXT = np.zeros((num_neurons, 1))
        self.epsilon = time_step
        self.somaV = self.U
        
        self.env.process(self.update())
    def update(self):
        while True:
            dUdt =  -(self.U - self.U_REST) + self.M_R * self.CURRENT_EXT
            dUdt = np.divide(dUdt,self.tau_membrane)
            self.U += dUdt * self.epsilon
            self.U[self.U > self.u_threshold] = self.u_reset
            self.somaV = np.concatenate([self.somaV, self.U], axis = 1)
            yield self.env.timeout(self.epsilon)

def stepCurrent(env, ensemble, current_amount, time_range):
    assert ensemble.CURRENT_EXT.shape == current_amount.shape
    assert len(time_range) == 2
    yield env.timeout(time_range[0])
    original_current = ensemble.CURRENT_EXT
    ensemble.CURRENT_EXT = current_amount
    yield env.timeout(time_range[1] - time_range[0])
    ensemble.CURRENT_EXT = original_current

sim_time = 10
initial_u = -50
current_amount = np.array([100, 5, 0, 10 ,10]).reshape(test_num_neurons,1)
env = simpy.Environment()


nn = Ensemble(env, initial_u)
env.process(stepCurrent(env, nn, current_amount, (2,5)))


env.run(until = sim_time)

fig, axes = plt.subplots(test_num_neurons, 1, sharex=True)
for idx,axis in enumerate(axes):
    axis.plot(np.arange(nn.somaV.shape[1])*epsilon, nn.somaV[idx,:] )
plt.show()