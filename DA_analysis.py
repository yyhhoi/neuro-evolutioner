import pandas as pd

from neuroevolutioner.Evolution import DA_Evolutioner, DA_Simulator


num_generations = 20
num_species = 1000
time_step = 0.0001

# evo = DA_Evolutioner("delayed_activation", num_generations, num_species, time_step)
# for i in range(num_generations):
#     evo.proliferate_one_generation(i)

evo_vis = DA_Simulator( project_name="delayed_activation", 
                        exp_results_dir = "experiment_results", 
                        vis_dir = "experiment_results/visualisation",
                        num_generations=num_generations,
                        num_species=num_species, 
                        time_step=time_step)
evo_vis.simulation_for_visualisation(15,585)
