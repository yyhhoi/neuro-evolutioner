from neuroevolutioner.Environments import Simulation
from neuroevolutioner.Ensembles import Ensemble_AdEx
from neuroevolutioner.utils import load_pickle, write_pickle
from neuroevolutioner.Genetics import TL_ParamsInitialiser, ConfigsConverter, TL_FitnessMeasurer
from neuroevolutioner.Experiments import TL_Experimenter
from neuroevolutioner.Probes import Probe
from glob import glob

import numpy as np
import os
import pandas as pd






def simulate_and_get_activity(data_dir, configs = None, gen_idx = 1, species_idx = 1, time_step = 0.0005):
    
    # Define storage directories
    os.makedirs(data_dir, exist_ok=True)
    activity_record_path = os.path.join(data_dir, "activity.csv")
    firing_rate_path = os.path.join(data_dir, "firing_rate.csv")
    stimuli_path = os.path.join(data_dir, "stimuli.csv")
    gene_save_path = os.path.join(data_dir, "gene.pickle")
    
    # Sample from gen template distributions to create configuration of the species
    params_initialiser = TL_ParamsInitialiser()
    if configs is None:
        configs = params_initialiser.sample_new_configs()
    num_neurons = configs["num_neurons"]
    anatomy_matrix, anatomy_labels = configs["anatomy_matrix"], configs["anatomy_labels"]

    # Initialise experimental paradigm
    exper = TL_Experimenter(num_neurons, anatomy_labels)

    # Initialise simulation environment and neuronal ensemble
    simenv = Simulation(exper.max_time, epsilon=time_step)
    ensemble = Ensemble_AdEx(simenv, num_neurons)
    ensemble.initialize_parameters(configs)
    
    # Initialise probing
    probe = Probe(
        num_neurons = num_neurons,
        activity_record_path = activity_record_path,
        stimuli_record_path = stimuli_path,
        gene_save_path = gene_save_path
    )
    
    # Simulation starts
    while simenv.sim_stop == False:
        time  = simenv.getTime()
        # Print progress
        print("\r{}/{}".format(time,exper.max_time), flush=True, end="")
        
        # Get current conditions and amount of external currents, given current time
        _, condition, label,  I_ext = exper.get_stimulation_info(time)

        # Apply current and update the dynamics
        ensemble.I_ext = I_ext * 1e-9
        ensemble.state_update()

        # Increment simulation environment
        simenv.increment()

        # Write out records
        probe.write_out_activity(time, condition, ensemble.firing_mask.get_mask().astype(int).astype(str))
        
    
    # Save genes
    print("\nSaving the genes")
    probe.save_gene(configs)


def proliferate_first_generation(project_results_dir, gen_idx=1, num_species=1000, time_step=0.0005):

    for species_idx in range(num_species):
        data_dir = os.path.join(project_results_dir, "generation_{}/species_{}/".format(gen_idx, species_idx))
        os.makedirs(data_dir, exist_ok=True)
        if os.path.isfile(os.path.join(data_dir, "gene.pickle")) :
            print("Generation {} and species {} exist. Skipped".format(gen_idx, species_idx))
        else:
            print("Generation: {} | Species: {}/{}".format(gen_idx, species_idx, num_species))
            simulate_and_get_activity(data_dir,  gen_idx=gen_idx, species_idx=species_idx, time_step=time_step)

def evaluate_generation_fitness(project_results_dir, gen_idx):
    generation_dir = os.path.join(project_results_dir, "generation_{}".format(gen_idx))
    hall_of_fame_path = os.path.join(project_results_dir, "hall_of_fame", "generation_{}.csv".format(gen_idx))
    all_species_dirs = sorted(glob(os.path.join(generation_dir, "*")))
    num_species = len(all_species_dirs)
    species_score_map = dict()
    species_idx_list, scores_list = [], []

    fh = open(hall_of_fame_path, "w")
    fh.write("species_idx,score\n")

    for i in range(num_species):

        activity_csv_path = os.path.join(generation_dir, "species_{}".format(i), "activity.csv")
        activity = pd.read_csv(activity_csv_path)
        
        # Evaluate the fitness score
        measurer = TL_FitnessMeasurer(activity)
        measurer.build_score_criteria()
        fitness_score = measurer.calc_fitness()
        species_idx_list.append(i)
        scores_list.append(fitness_score)
        fh.write("%d,%0.4f\n" % (i, fitness_score))
        print("Evaluated {}/{}: Score = {}".format(i, num_species, fitness_score))
    fh.close()
    species_score_map["species_idx"], species_score_map["score"] = species_idx_list, scores_list
    score_df = pd.DataFrame(species_score_map)
    return score_df

# def evolute_one_generation(projectscore_df)


    


if __name__ == "__main__":

    project_results_dir = "experiment_results/time_learning"
    
    proliferate_first_generation(project_results_dir, 2)
    
    # score_df = evaluate_generation_fitness(project_results_dir, 1)
    # print(score_df.head())
    
    # # Debug of times and conditions list
    # times_list, conditions_list, conditions = gen_condition_time_list()
    # for time,cond in zip(times_list, conditions_list):
    #     print("time: {} | cond: {}".format(time, cond))

    # # Debug of experimenter
    # configs = get_SG_configs()
    # num_neurons = configs["Meta"]["num_neurons"]
    # anatomy_matrix, anatomy_labels = SG_anatomy()
    # exper = Experimenter(num_neurons, anatomy_labels)
    # time_step = 0.1
    # simenv = Simulation(exper.max_time, epsilon=time_step)
    # while simenv.sim_stop == False:
    #     time  = simenv.getTime()
    #     # Get current conditions and amount of external currents, given current time
    #     _, condition, label,  I_ext = exper.get_stimulation_info(time)
    #     print("time: %0.3f | Condition: %s | label: %s\n I_ext_sen:\n%s\n I_ext_act:\n%s" %(time, condition, label,
    #                                                                                         str(list(I_ext[0:anatomy_labels["sensory2"]])),
    #                                                                                         str(list(I_ext[anatomy_labels["brain"]:anatomy_labels["class2"]]))
    #                                                                                         ))
    #     simenv.increment()