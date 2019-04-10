import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
from neuroevolutioner.Genetics import DA_FitnessMeasurer
import numpy as np



gen_idx = 0
time_step = 0.0001
base_dir = "experiment_results/delayed_activation/generation_{}".format(gen_idx)
HOF_path = os.path.join(base_dir, "hall_of_fame.csv")
HOF_df = pd.read_csv(HOF_path)
HOF_df = HOF_df.sort_values(by="score", ascending=False)

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
    firing_np = np.array(firing.iloc[:, 2:])
    print("NP's Sum:\n", np.sum(firing_np, axis=0))
    print("Action's firing indexes:\n", np.where(firing_np[:, 9] > 0))
    mea = DA_FitnessMeasurer(firing, time_step)
    fitness = mea.calc_fitness()
    print("Fitness = {}".format(fitness))
    fig, ax = plt.subplots()
    ax.imshow(firing_np, aspect="auto")
    plt.show()