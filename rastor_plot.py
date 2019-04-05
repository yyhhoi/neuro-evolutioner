import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
from neuroevolutioner.genetics import TL_FitnessMeasurer
import numpy as np

base_dir = "experiment_results/time_learning/generation_1"
all_species_paths = glob(os.path.join(base_dir, "*"))

stimuli_dict = {
    "train_S": [2],
    "train_A": [11],
    "test_S": [2]
}

def produce_stimuli_raster(firing, stimuli_dict):
    stimuli_np = np.zeros(firing.shape)

    for condition_key in stimuli_dict.keys():
        mask = (firing["condition"] == condition_key)
        stimuli_np[mask, stimuli_dict[condition_key]] = 1
    stimuli_np = stimuli_np * np.array(firing["time"]).reshape(-1, 1)
    stimuli_np = stimuli_np[np.sum(stimuli_np[:, 2:], axis = 1) > 0]
    return stimuli_np
        


for idx, species_path in enumerate(all_species_paths):
    firing_data_path = os.path.join(species_path, "activity.csv")
    firing = pd.read_csv(firing_data_path)

    # Calc fitness
    fitnesser = TL_FitnessMeasurer(firing)
    fitnesser.build_score_criteria()
    fitness_score = fitnesser.calc_fitness()
    
    # Convert the numpy
    stimuli_np = produce_stimuli_raster(firing, stimuli_dict)
    firing_np = np.array(firing)
    
    # Convert to raster plot format
    firing_np = firing_np[:, 2:] * firing_np[:, 0].reshape(-1,1)

    # filter out rows with all silent neurons
    firing_np_sum = np.sum(firing_np, axis = 1)
    firing_np = firing_np[firing_np_sum >0, :]
    print("Firing's Shape = {}\nStimuli's Shape = {}".format(firing_np.shape, stimuli_np.shape))
    
    # For stimuli, we only need a subset of them
    np.random.seed(90)
    ran_vec = np.random.choice(stimuli_np.shape[0], size = (200,))
    stimuli_np = stimuli_np[ran_vec, :]

    if firing_np.shape[0] > 500:
        np.random.seed(90)
        ran_vec = np.random.choice(firing_np.shape[0], size = (500,))
        firing_np = firing_np[ran_vec, :]
    fig, ax = plt.subplots()
    ax.eventplot(stimuli_np[:, 2:].T, alpha=0.1)
    ax.eventplot(firing_np.T, color = "r")
    fig.suptitle("Fitness = %0.3f" % fitness_score)

    
    plt.show()