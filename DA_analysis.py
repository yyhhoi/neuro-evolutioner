import pandas as pd

from neuroevolutioner.Evolution import DA_Evolutioner, DA_Simulator
from neuroevolutioner.DataProcessing import DA_data_processing
from neuroevolutioner.Visualisation import Visualiser_wrapper

project_name = "delayed_activation"
num_generations = 20
num_species = 1000
time_step = 0.0001

# ======================= Evolution =========================================
# evo = DA_Evolutioner(project_name, exp_results_dir = "experiment_results", num_generations, num_species, time_step)
# for i in range(num_generations):
#     evo.proliferate_one_generation(i)

# ======================= Simulation for visualisation =========================================
selected_gen = 15
selected_species = 585
vis_dir = "experiment_results/visualisation"
# evo_vis = DA_Simulator( project_name=project_name, 
#                         exp_results_dir = "experiment_results", 
#                         vis_dir = vis_dir,
#                         num_generations=num_generations,
#                         num_species=num_species, 
#                         time_step=time_step)
# evo_vis.simulation_for_visualisation(selected_gen,selected_species)

# ======================= preprocessing =========================================
# da_pro = DA_data_processing(project_nam5e, vis_dir, time_step)
# da_pro.produce_firing_rate(selected_gen, selected_species)

# ======================= Generate graphs ======================================
import warnings
warnings.filterwarnings("error")
viser = Visualiser_wrapper(project_name,
                           vis_dir,
                           selected_gen,
                           selected_species,
                           time_step)

viser.initialise()
# viser.generate_graphs()
viser.combine_graphs_to_video()