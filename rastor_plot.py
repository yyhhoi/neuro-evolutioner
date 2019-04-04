import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

base_dir = "experiment_results/time_learning/generation_1"
all_species_paths = glob(os.path.join(base_dir, "*"))


for idx, species_path in enumerate(all_species_paths):
    firing_data_path = os.path.join(species_path, "activity.csv")
    stimuli_data_path = os.path.join(species_path, "stimuli.csv")
    firing = pd.read_csv(firing_data_path)
    stimuli = pd.read_csv(stimuli_data_path)
    
    # Convert the numpy
    firing_np = np.array(firing)
    stimuli_np = np.array(stimuli)
    
    # Convert to raster plot format
    firing_np = firing_np[:, 1:] * firing_np[:, 0].reshape(-1,1)
    stimuli_np_with_neurons_only = stimuli_np[:, 3:]
    stimuli_np_with_neurons_only[stimuli_np_with_neurons_only > 0] = 1
    stimuli_np = stimuli_np_with_neurons_only * stimuli_np[:, 0].reshape(-1,1)
    
    # filter out rows with all silent neurons
    firing_np_sum = np.sum(firing_np, axis = 1)
    stimuli_np_sum = np.sum(stimuli_np, axis = 1)
    firing_np = firing_np[firing_np_sum >0, :]
    stimuli_np = stimuli_np[stimuli_np_sum >0, :]
    print("Firing's Shape = {}\nStimuli's Shape = {}".format(firing_np.shape, stimuli_np.shape))
    
    # For stimuli, we only need a subset of them
    np.random.seed(90)
    ran_vec = np.random.choice(stimuli_np.shape[0], size = (500,))
    stimuli_np = stimuli_np[ran_vec, :]
    if firing_np.shape[0] > 2000:
        np.random.seed(90)
        ran_vec = np.random.choice(firing_np.shape[0], size = (500,))
        firing_np = firing_np[ran_vec, :]
    fig, ax = plt.subplots()
    ax.eventplot(stimuli_np.T, alpha=0.1)
    ax.eventplot(firing_np.T, color = "r")
    
    plt.show()