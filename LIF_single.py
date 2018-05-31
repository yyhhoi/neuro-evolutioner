import numpy as np
import simpy
import matplotlib.pyplot as plt
import pdb
from parameters import *
class Network(object):
    # 2D map [presynaptic, postsynaptic] for storing temporary data of synaptic currents and weights of connections
    syncurrent_map = np.zeros((max_neurons_num,max_neurons_num))
    weights_map = np.zeros((max_neurons_num, max_neurons_num))
    processes_monitor = []
    def updateSynCurrentMap(self, pre_id, post_id, current):
        __class__.syncurrent_map[pre_id, post_id] = current
    def incrementWeightsMap(self, pre_id, post_id, delta_weight):
        __class__.weights_map[pre_id, post_id] += delta_weight
    def getPostCurrent(self, post_id):
        return np.sum(__class__.syncurrent_map[:, post_id])
    def getWeight(self, pre_id, post_id):
        return __class__.weights_map[pre_id, post_id]
class Connection(Network):
    def __init__(self, env, pre_neuron, post_neuron, tau_syn = 1, rev_potential = 0,g_syn = 1,
                 weight_ini = 1, x1_tau = 1, y1_tau = 1, y2_tau = 5,
                 weight_lr = 0.01, weight_decay_rate = 5, max_weight = 2, learning_enabled = True):
        self.env = env
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.x1_record, self.y1_record, self.y2_record, self.weight_record, self.syn_process_monitor = [], [], [], [], []
        # Activate the increment
        self.LTP_check = 0
        self.LTD_check = 0

        # Initialize events
        self.LTP_allowed = self.env.event()
        self.LTD_allowed = self.env.event()

        # Initialise the weight of the connection
        super(Connection, self).weights_map[self.pre_neuron.id, self.post_neuron.id] = weight_ini
        
        # Initialise parameters for the synapses
        self.tau_syn, self.rev_potential, self.g_syn = tau_syn, rev_potential, g_syn
        
        # Last spike time starts from negative infinity
        self.last_pre_spike_time = -9999
        self.last_post_spike_time = -9999
        
        # Initialise necessary processes for (1) spike detection and (2)update of synaptic current
        self.pre_spike_detector = self.env.process(self.preSpikingListener())
        self.post_spike_detector = self.env.process(self.postSpikingListener())
        self.syncurrent_updater = self.env.process(self.synapticCurrent())
        
        # Enable synaptic plasticity
        if learning_enabled == True:
            self.weight_incrementer = self.env.process(self.weightIncrementer(x1_tau, y1_tau, y2_tau,
                                                                            weight_lr, weight_decay_rate, max_weight))
            self.env.process(self.LTP_activated())
            self.env.process(self.LTD_activated())
    def preSpikingListener(self):
        while True:
            yield self.pre_neuron.spikeOccur
            self.last_pre_spike_time = self.env.now
            self.LTD_allowed.succeed()
            self.LTD_allowed = self.env.event()
    def postSpikingListener(self):
        while True:
            yield self.post_neuron.spikeOccur
            self.last_post_spike_time = self.env.now    
            self.LTP_allowed.succeed()
            self.LTP_allowed = self.env.event()
    def synapticCurrent(self):
        while True:
            decay_kernel = np.exp(-(self.env.now - self.last_pre_spike_time)/self.tau_syn)
            gate = self.g_syn * decay_kernel * super(Connection, self).getWeight(self.pre_neuron.id, self.post_neuron.id)
            syncurrent_this_connection = gate * (self.post_neuron.u - self.rev_potential)
            super(Connection, self).updateSynCurrentMap( self.pre_neuron.id, self.post_neuron.id, syncurrent_this_connection)
            yield self.env.timeout(epsilon)
    def weightIncrementer(self, x1_tau, y1_tau, y2_tau, weight_lr, weight_decay_rate, max_weight):
        while True:
            weight_this = super(Connection, self).getWeight(self.pre_neuron.id, self.post_neuron.id)
            x1_trace = np.exp(-(self.env.now - self.last_pre_spike_time)/x1_tau)
            y1_trace = np.exp(-(self.env.now - self.last_post_spike_time)/y1_tau)
            y2_trace = np.exp(-(self.env.now - self.last_post_spike_time)/y2_tau)
            # Record x1, y1, y2, weight 's data for analysis and plotting
            self.weight_record.append(weight_this)
            self.x1_record.append(x1_trace)
            self.y1_record.append(y1_trace)
            self.y2_record.append(y2_trace)
            delta_LTP = x1_trace * y2_trace
            delta_LTD = y1_trace*0.5
            # LTP
            if self.LTP_check == 1:
                self.LTP_check = 0
                delta_weight = weight_lr * (max_weight - weight_this) * (delta_LTP)
                super(Connection, self).incrementWeightsMap(self.pre_neuron.id, self.post_neuron.id, delta_weight)
            # LTD
            if self.LTD_check == 1:
                self.LTD_check = 0
                delta_weight = weight_lr * (max_weight - weight_this) * (- delta_LTD)
                super(Connection, self).incrementWeightsMap(self.pre_neuron.id, self.post_neuron.id, delta_weight)
            # Natural decay
            delta_weight = - weight_decay_rate * weight_this
            super(Connection, self).incrementWeightsMap(self.pre_neuron.id, self.post_neuron.id, delta_weight)
            yield self.env.timeout(epsilon)
    def LTP_activated(self):
        while True:
            yield self.LTP_allowed
            self.LTP_check = 1
            yield self.env.timeout(epsilon)
    def LTD_activated(self):
        while True:
            yield self.LTD_allowed
            self.LTD_check = 1
            yield self.env.timeout(epsilon)
        

class Neuron(Network):
    count = 0 
    def __init__(self, env, name = "null"):
        # Parameters for passive membrane, threshold, reset, ... etc
        self.tau_membrane, self.u, self.u_rest, self.R, self.current = 10, -70, -70, 1, 0
        self.u_threshold, self.u_reset, self.syncurrent = 70, -80, 0
        # Parameters for Izhikevich model
        self.w, self.b, self.a, self.tau_w, self.u_critical = 0, 50, 0 , 5, -40
        # Recording / Neuron's information
        self.somaV, self.spike_record  = [], []
        self.connected_to, self.connected_by = [], []
        self.name = name
        self.id = self.__class__.count
        self.__class__.count += 1
        self.env = env
        self.action = env.process(self.stateUpdate())
        self.spikeOccur = env.event()
    def stateUpdate(self):
        while True:
            self.somaV.append(self.u)
            self.syncurrent = super(Neuron,self).getPostCurrent(self.id)
            # Main body of Differential Equations
            dudt =  (self.u - self.u_rest) * (self.u - self.u_critical ) + self.R * self.current - self.syncurrent - self.R * self.w
            # Integrate and fire model
            # dudt =  -(self.u - self.u_rest) + self.R * self.current - self.syncurrent
            dudt = dudt/self.tau_membrane 
            dwdt = self.a * (self.u - self.u_rest) - self.w
            dwdt = dwdt/self.tau_w

            self.u += dudt * epsilon
            self.w += dwdt * epsilon
            # ^ Main body of Differential Equations
            
            if self.u > self.u_threshold:
                self.spike_record.append(( self.env.now, self.u))
                self.u = self.u_reset
                self.spikeOccur.succeed()
                self.spikeOccur = env.event()
                # After-spike update of Izhikevich model
                self.w += self.b
            yield self.env.timeout(epsilon)

    def getSpikeRecord(self):
        if len(self.spike_record) == 0:
            return 0
        else:
            y = np.array(self.spike_record)
            spike_time = y[:,0]
            potential = y[:,1]
            return (spike_time, potential)
        

def impulseU(env, neuron, time_train):
    time_train = np.append(0,time_train)
    for i in range(len(time_train)-1):
        yield env.timeout(time_train[i+1] - time_train[i])
        neuron.u += impulse_amount

def stepCurrent(env, neuron, current_amount, time_range):
    yield env.timeout(time_range[0])
    original_current = neuron.current
    neuron.current = current_amount
    yield env.timeout(time_range[1] - time_range[0])
    neuron.current = original_current

def plot_soma(time, neurons_list):
    n = len(neurons_list)
    if n >1:
        f, axes = plt.subplots(n, sharex = True)
        for idx,neuron in enumerate(neurons_list):
            axes[idx].plot(time,neuron.somaV)
            if neuron.getSpikeRecord() != 0:
                # print("Neuron_%d Spike num: %d" % (neuron.id, len(neuron.getSpikeRecord()[0])))
                axes[idx].scatter(neuron.getSpikeRecord()[0], neuron.getSpikeRecord()[1], c='r')
            if neuron.name == "null":
                
                axes[idx].set_title("neuron_" + str(neuron.id))
            else:
                axes[idx].set_title(neuron.name)
        return axes
    elif n == 1:
        f, axes = plt.subplots(n, sharex = True)
        for idx,neuron in enumerate(neurons_list):
            axes.plot(time,neuron.somaV)
            if neuron.getSpikeRecord() != 0:
                axes.scatter(neuron.getSpikeRecord()[0], neuron.getSpikeRecord()[1], c='r')
            if neuron.name == "null":
                
                axes.set_title("neuron_" + str(neuron.id))
            else:
                axes.set_title(neuron.name)
        return axes
    



# Set up simulation environment
env = simpy.Environment()

# Create neurons (processes are initialised within instance)
in1 = Neuron(env, name = "input 1")
out1 = Neuron(env, name = "from 1")

# Applying step current to the input neurons
# env.process(stepCurrent(env, in1, 100, (5,20)))
# env.process(stepCurrent(env, neuron_in2, 100, (5,20)))
env.process(impulseU(env, in1, [5, 8]))
env.process(impulseU(env, out1, [5.1, 6]))
# Build connection. (env, presynaptic neuron, postsynaptic neuron)
in1_out1 = Connection(env, in1, out1, rev_potential = 60, g_syn = 1, weight_ini = 0,
                     x1_tau=5, y1_tau=5, y2_tau=5, weight_lr=0.3, weight_decay_rate = 0,  learning_enabled=True)

neurons_list = [in1, out1]
# Run the simulation
sim_time = 20
env.run(until = sim_time)

# Plot the soma's voltage of the neurons
time = np.arange(len(neurons_list[0].somaV)) * epsilon
# neurons_list = [neuron_in1, neuron_in2, neuron_out1, neuron_out2]
print("Total processes: %d" % len(Network.processes_monitor))

axes = plot_soma(time, neurons_list)
if len(axes) == 0:
    axes.set_xticks(np.arange(0, sim_time, step = 1))
    # axes.set_ylim(-100,100)
else:
    for i in range(len(axes)):
        axes[i].set_xticks(np.arange(0, sim_time, step = 1))
        # axes[i].set_ylim(-100,100)

f, axes2 = plt.subplots(1)
axes2.plot(time, in1_out1.x1_record, label="x1")
axes2.plot(time, in1_out1.y1_record, label="y1")
axes2.plot(time, in1_out1.y2_record, label="y2")
axes2.plot(time, in1_out1.weight_record, label="weight")
axes2.legend()

plt.show()
