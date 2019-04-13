import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
from neuroevolutioner.Genetics import DA_FitnessMeasurer
import numpy as np
import pdb
gen_idx = 26
time_step = 0.0005
base_dir = "experiment_results/delayed_activation/generation_{}".format(gen_idx)
HOF_path = os.path.join(base_dir, "hall_of_fame.csv")
HOF_df = pd.read_csv(HOF_path)
HOF_df = HOF_df.sort_values(by="score", ascending=False)

stimuli_dict = {
    "S": [2]
}

def produce_stimuli_raster(firing, stimuli_dict):
    stimuli_np = np.zeros(firing.shape)

    for condition_key in stimuli_dict.keys():
        mask = (firing["condition"] == condition_key)
        stimuli_np[mask, stimuli_dict[condition_key]] = 1
    stimuli_np = stimuli_np * np.array(firing["time"]).reshape(-1, 1)
    stimuli_np = stimuli_np[np.sum(stimuli_np[:, 2:], axis = 1) > 0]
    return stimuli_np
        

for idx in range(HOF_df.shape[0]):
    if HOF_df.iloc[idx, 0] != gen_idx:
        continue
    print(HOF_df.iloc[idx,:])
    species_idx = HOF_df.iloc[idx, 1]
    # fitness_calc = HOF_df[idx]["score"]
    

    species_path = os.path.join(base_dir, "species_{}".format(species_idx))
    firing_data_path = os.path.join(species_path, "activity.csv")
    try:
        firing = pd.read_csv(firing_data_path)
    except FileNotFoundError:
        continue
    
    # Calc fitness
    fitnesser = DA_FitnessMeasurer(firing, time_step)
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

    fig, ax = plt.subplots()
    ax.eventplot(stimuli_np[:, 2:].T, alpha=0.1)
    ax.eventplot(firing_np.T, color = "r")
    fig.suptitle("Generation: %d | Species: %d | Fitness = %0.3f" % (gen_idx, species_idx , fitness_score))
    
    plt.show()