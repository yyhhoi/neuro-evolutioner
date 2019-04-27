import numpy as np
import matplotlib.pyplot as plt
from neuroevolutioner.Ensembles import Ensemble_AdEx
from neuroevolutioner.Environments import Simulation
import os

# Raise exception for runtime warning because we don't want numerical overflow.
import warnings
warnings.filterwarnings("error")

# Hard-coded parameters
num_neurons = 2
I_ext_amount = 1.2e-10
u_threshold = -50e-3
project_name = "simple_association"
time_step = 0.001
simulation_time = 5

# Configuration of neuronal/synaptic dynamics parameters
configs = {
    # (Required) Parameters for LIF neurons
    "u_rest": np.ones(num_neurons) * -70e-3,
    "r_m": np.ones(num_neurons) * 500e6,
    "tau_m": np.ones(num_neurons) * 180e-3,
    "u_threshold": np.ones(num_neurons) * u_threshold,
    "u_reset": np.ones(num_neurons) * -55e-3,
    # (Required) Parameters of AdEx neurons
    "sharpness": np.ones(num_neurons) * 2e-3,
    "tau_w": np.ones(num_neurons) * 100e-3,
    "a": np.ones(num_neurons) * 32e-10,
    "b": np.ones(num_neurons) * 5e-10,

    # (Required) Synaptic current's dynamics
    "tau_syn": np.ones((num_neurons, num_neurons)) * 1e-1,
    "E_syn": np.eye(num_neurons)[::-1] * 0,  # Reversal potential. All are excitatory, i.e. E_syn=0
    "g_syn_constant": np.eye(num_neurons)[::-1] * 1e-7,

    # Synaptic configuration.
    "anatomy_matrix": np.eye(num_neurons)[::-1],  # Anatomical constraints, 1=connection possible, 0 otherwise
    "types_matrix": np.eye(num_neurons)[::-1],  # 1 = excitatory, 0 = inhibitory
    "w": np.eye(num_neurons)[::-1] * 0,  # Initially, weights = 0
    "w_max": np.eye(num_neurons)[::-1] * 5, # Maximum weights. Excceeding this values will be clipped

    # (Required)  Synaptic plasticity. Here, only simple triplet STDP is used.
    # Other dynamics such as Heterosynaptic decay/ homeostatic dynamics / special inihibitory dynamics are disabled.for
    # Reference of Triplet STDP: https://www.ncbi.nlm.nih.gov/pubmed/16988038
    # Traces for triplet STDP
    "tau_LTP": np.ones(num_neurons) * 8e-2,
    "tau_LTP_slow": np.ones(num_neurons) * 3e-1,
    "tau_LTD": np.ones(num_neurons) * 1e-2,
    # Contribution of STDP to weights is unrealistically high here,
    # because we want to observe the weight increase in only 2 neurons
    "A": np.eye(num_neurons)[::-1] * 2e7,


    # From here below, all dynamics are disabled with dummy parameters.
    # Dynamics rules below are the reproduction of Zenke's, Agnes's and Gerstner's work in 2015 (https://www.nature.com/articles/ncomms7922)
    # which includes:
    #   1. Heterosynaptic dynamics that penalise the weight that cause over-firing in post-synatic neurons, which has a
    #       bistable property
    #   2. Neuro-transmitter induced-plasticity that increase the postsynaptic weight with every delivery of spike
    #   3. Slow homeostatic regulation of LTD: Reduce the effect of LTD if the network remains rather silent for too long
    #   4. Inhibitory synaptic dynamics
    # Heterosynaptic dynamics
    "beta": np.eye(num_neurons)[::-1] * 0,
    "w_p": np.eye(num_neurons)[::-1] * 0.5,
    "P": np.eye(num_neurons)[::-1] * 2,
    "tau_cons": np.ones((num_neurons, num_neurons)) * 1,
    # Neuro-transmitter excitation disabled
    "transmitter_constants": np.eye(num_neurons)[::-1] * 0,
    # Slow homeostatic dynamics disabled
    "tau_hom": np.ones((num_neurons, num_neurons)) * 9e3,
    "tau_ht": np.ones(num_neurons) * 9e3,
    # Inhibtory synaptic dynamics disabled
    "eta": np.eye(num_neurons)[::-1] * 0,
    "gamma": np.eye(num_neurons)[::-1] * 0,
    "tau_H": np.ones((num_neurons, num_neurons)) * 9e3
}

# Convert time to experiment conditions
def time2cond(time):
    if (0 < time < 0.5) or (3.5 < time < 4):
        return "neuron1"
    elif (1 < time < 1.5):
        return "neuron2"
    elif (2 < time < 3):
        return "both"
    else:
        return "rest"


# Convert experiment conditions to the amount of external current
def cond2current(cond):
    if cond == "rest":
        return np.zeros(num_neurons)
    elif cond == "both":
        return np.ones(num_neurons) * I_ext_amount
    elif cond == "neuron1":
        return np.array([1, 0]) * I_ext_amount
    elif cond == "neuron2":
        return np.array([0, 1]) * I_ext_amount



# initialise simulation environment
simenv = Simulation(simulation_time, time_step)

# Initialise the list of states to be recorded
recorder = {
    "neuron_u": [], # first and second neurons
    "weights": [], # [times, 0] = from first to second. [times, 1] = second to first
    "syn_currents": [],
    "external_currents": [],
    "LTP_trace": [],
    "LTP_slow": [],
    "LTD_trace": [],
    "w(adex)": []
}
firing_list = []
ensemble = Ensemble_AdEx(simenv, num_neurons)
ensemble.initialize_parameters(configs)

while simenv.sim_stop == False:
    time = simenv.getTime()

    # Print progress
    print("\rSimulation time: %f/%f" % (time, simulation_time), flush=True, end="")

    # Set the external current
    ensemble.I_ext = cond2current(time2cond(time))

    # Record the neuronal states
    firing_list.append(ensemble.firing_mask.get_mask())
    recorder["neuron_u"].append(ensemble.u.copy())
    weights = ensemble.Weights.get_weights().copy()
    recorder["weights"].append(np.array([weights[0,1], weights[1, 0]]))
    recorder["syn_currents"].append(ensemble.synaptic_currents.copy())
    recorder["external_currents"].append(ensemble.I_ext.copy())
    recorder["LTP_trace"].append(ensemble.Weights.z_LTP.get_trace().copy())
    recorder["LTP_slow"].append(ensemble.Weights.z_LTP_slow.get_trace().copy())
    recorder["LTD_trace"].append(ensemble.Weights.z_LTD.get_trace().copy())
    recorder["w(adex)"].append(ensemble.w.copy())

    # Increment the simulation
    ensemble.state_update()
    simenv.increment()


# Plotting
arr_fire = np.array(firing_list)
fire_idx = np.where(arr_fire == 1)
fire_time_np = np.array(fire_idx[0])* time_step

fig, ax = plt.subplots(len(recorder.keys()), sharex=True, figsize=(18, 10))
for idx, key in enumerate(recorder.keys()):
    arr = np.asarray(recorder[key])
    # If it is plotting the soma potential, plot also when it fires.
    if key == "neuron_u":
        for i in range(fire_time_np.shape[0]):
            plot_arr = np.array([
                [fire_time_np[i], u_threshold],
                [fire_time_np[i], u_threshold + 10e-3]
            ])
            ax[idx].plot(plot_arr[:, 0], plot_arr[:, 1], color="red")
    # Plots neuronal/synaptic states of interest
    ax[idx].plot(np.arange(arr.shape[0]) * time_step, arr[:, 0], label=key+"_1")
    ax[idx].plot(np.arange(arr.shape[0]) * time_step, arr[:, 1], label=key + "_2")
    ax[idx].legend()

fig.suptitle("Neuronal states during simulation with AdEx Neurons and Triplet STDP rules")
plt.savefig(os.path.join("figs/simple_association.png"))
plt.show()


