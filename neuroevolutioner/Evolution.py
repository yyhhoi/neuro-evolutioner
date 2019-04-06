from neuroevolutioner.Environments import Simulation
from neuroevolutioner.Ensembles import Ensemble_AdEx
from neuroevolutioner.utils import load_pickle, write_pickle
from neuroevolutioner.Genetics import TL_ParamsInitialiser, ConfigsConverter, TL_FitnessMeasurer, crossover, mutation
from neuroevolutioner.Experiments import TL_Experimenter
from neuroevolutioner.Probes import Probe

from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
from glob import glob

class Evolutioner(ABC):
    def __init__(self, project_name, num_generations=10, num_species=1000, time_step = 0.005):
        self.project_name, self.num_gens, self.num_species = project_name, num_generations, num_species
        self.activity_results_filename, self.gene_results_filename, self.HOF_filename = "activity.csv", "gene.pickle", "hall_of_fame.csv"
        self.winners_filename = "winners.csv"
        self.proj_results_dir = os.path.join("experiment_results", project_name)
        self.time_step = time_step
        

    def proliferate_one_generation(self, gen_idx):
        self._initialise_HOF(gen_idx)
        for species_idx in range(self.num_species):
            if os.path.isfile(os.path.join(self.get_species_dir(gen_idx, species_idx), self.gene_results_filename)) :    
                print("Generation {} and species {} exist. Skipped".format(gen_idx, species_idx))
            else:
                if gen_idx == 0:
                    configs = None
                else:
                    print("Sampling new configs from winners")
                    configs = self._sample_configs_from_winners(gen_idx-1) 
                print("Generation: {} | Species: {}/{}".format(gen_idx, species_idx, self.num_species))
                self._single_simulation(gen_idx=gen_idx, species_idx=species_idx, configs=configs)
        # self.evaluate_one_generation(gen_idx) # Produce hall_of_frame in gen_dir
        self.select_winners(gen_idx)
    def _single_simulation(self, gen_idx, species_idx, configs=None):
        
        # Define paths
        offspring_results_dir = self.get_species_dir(gen_idx, species_idx)
        os.makedirs(offspring_results_dir, exist_ok=True)
        activity_record_path = os.path.join(offspring_results_dir, self.activity_results_filename)
        gene_save_path = os.path.join(offspring_results_dir, self.gene_results_filename)

        # custom genes or import from existing one
        if configs is None:
            configs = self._sample_new_configs()
        num_neurons = configs["num_neurons"]
        _, anatomy_labels = configs["anatomy_matrix"], configs["anatomy_labels"]

        # Initialise experimental paradigm
        exper = self._initialise_experimenter(num_neurons, anatomy_labels)

        # Initialise simulation environment and neuronal ensemble
        simenv = Simulation(exper.max_time, epsilon=self.time_step)
        ensemble = Ensemble_AdEx(simenv, num_neurons)
        ensemble.initialize_parameters(configs)
        
        # Initialise probing
        probe = Probe(
            num_neurons = num_neurons,
            activity_record_path = activity_record_path,
            gene_save_path = gene_save_path
        )

        # Simulation starts
        self._loop_simulate(simenv, ensemble, exper, probe)

        # Get fitness score and print
        activity = pd.read_csv(self.get_activity_path(gen_idx, species_idx))
        fitness_score = self._calc_fitness_score(activity)
        self._write_to_HOF(gen_idx, species_idx, fitness_score)
        print("Fitness: %0.4f"%(fitness_score))

        # Save genes
        probe.save_gene(configs)

    def _initialise_HOF(self, gen_idx):
        HOF_path = self.get_HOF_path(gen_idx)
        if os.path.isfile(HOF_path) is not True:
            if gen_idx == 0:
                with open(HOF_path, "w") as fh:
                    fh.write("gen_idx,species_idx,score\n")
            else:
                # Maintain the winners in previous generation in the current Hall of Fame
                previous_winners_df = pd.read_csv(self.get_winners_path(gen_idx-1))
                previous_winners_df.to_csv(self.get_HOF_path(gen_idx), index=False)
        else:
            pass

    def _write_to_HOF(self, gen_idx, species_idx, fitness_score):
        with open(self.get_HOF_path(gen_idx), "a") as fh_a:
            fh_a.write("%d,%d,%0.4f\n" % (gen_idx, species_idx, fitness_score))

    # def evaluate_one_generation(self, gen_idx):
        
    #     # Sample all species' dirs
    #     all_species_dirs = sorted(glob(os.path.join(self.get_generation_dir(gen_idx), "*/")))
    #     num_species = len(all_species_dirs)

    #     # Write Record headers
    #     if gen_idx == 0:
    #         with open(self.get_HOF_path(gen_idx), "w") as fh:
    #             fh.write("gen_idx,species_idx,score\n")
    #     else:
    #         # Maintain the winners in previous generation in the current Hall of Fame
    #         previous_winners_df = pd.read_csv(self.get_winners_path(gen_idx-1))
    #         previous_winners_df.to_csv(self.get_HOF_path(gen_idx), index=False)
        
    #     # Loop through all species, calculate fitness and write to record.
    #     for i in range(num_species):
    #         generation_dir = os.path.join(self.proj_results_dir, "generation_{}".format(gen_idx))
    #         activity_csv_path = os.path.join(generation_dir, "species_{}".format(i), "activity.csv")
    #         activity = pd.read_csv(activity_csv_path)
            
    #         # Evaluate the fitness score
    #         fitness_score = self._calc_fitness_score(activity)
    #         with open(self.get_HOF_path(gen_idx), "a") as fh_a:
    #             fh_a.write("%d,%d,%0.4f\n" % (gen_idx, i, fitness_score))
    #         print("Evaluated Gen:{} | Species: {}/{} | Score: {}".format(gen_idx, i, num_species, fitness_score))
        
    def select_winners(self, gen_idx, fraction=0.1):
        num_winners_to_select = int(self.num_species * fraction)
        HOF_df = pd.read_csv(self.get_HOF_path(gen_idx))
        HOF_df = HOF_df.sort_values(by="score", ascending=False)
        HOF_df_selected = HOF_df.iloc[0:num_winners_to_select, ]
        HOF_df_selected.to_csv(self.get_winners_path(gen_idx), index=False)

    

    def _get_winners_idx(self, gen_idx):
        winners_df = pd.read_csv(self.get_winners_path(gen_idx))
        winners_idx_np = np.array(winners_df[["gen_idx", "species_idx"]])
        
        return winners_idx_np

    def _sample_configs_from_winners(self, gen_idx):
        # Get winners and sample two from them
        winners_idx_np = self._get_winners_idx( gen_idx) # ndarray with shape (num_winners, 2)
        ran_vec = np.random.permutation(winners_idx_np.shape[0])
        winner1, winner2 = winners_idx_np[ran_vec,:][0:2, :]

        # Load the configs according to the winner's generaiton and species id
        configs1 = load_pickle(self.get_gene_results_path(winner1[0], winner1[1]))
        configs2 = load_pickle(self.get_gene_results_path(winner2[0], winner2[1]))

        # Convert configs to chromosome, do crossover and mutation
        converter1, converter2 = ConfigsConverter(), ConfigsConverter()
        gene1, gene2 = converter1.configs2gene(configs1), converter1.configs2gene(configs2)
        chromosome1, chromosome2 = gene1["gene"]["chromosome"], gene2["gene"]["chromosome"]
        c_chrom1, c_chrom2 = crossover(chromosome1, chromosome2)
        mutated_chrom = mutation(c_chrom1)
        gene1["gene"]["chromosome"] = mutated_chrom
        new_config = converter1.gene2configs(gene1)
        return new_config

    def _create_config(self, gen_idx):
        # Produce configs
        if gen_idx == 0:
            configs = None
        else:
            configs = self._sample_configs_from_winners(gen_idx-1) 
        return configs

    @abstractmethod
    def _sample_new_configs(self):
        # params_initialiser = TL_ParamsInitialiser()
        # configs = params_initialiser.sample_new_configs()
        return None
    @abstractmethod
    def _calc_fitness_score(self, activity):
        # measurer = TL_FitnessMeasurer(activity)
        # measurer.build_score_criteria()
        # fitness_score = measurer.calc_fitness()
        return None
    @abstractmethod
    def _initialise_experimenter(self, num_neurons, anatomy_labels):
        # exper = TL_Experimenter(num_neurons, anatomy_labels)
        return None


    @staticmethod
    def _loop_simulate(simenv, ensemble, exper, probe):
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
    

    def get_generation_dir(self, gen_idx):
        return os.path.join(self.proj_results_dir, "generation_{}".format(gen_idx))
    
    def get_species_dir(self, gen_idx, species_idx):
        return os.path.join(self.proj_results_dir, "generation_{}".format(gen_idx), "species_{}".format(species_idx))
    def get_gene_results_path(self,gen_idx, species_idx):
        return os.path.join(self.proj_results_dir, "generation_{}".format(gen_idx), "species_{}".format(species_idx), self.gene_results_filename)

    def get_activity_path(self, gen_idx, species_idx):
        return os.path.join(self.proj_results_dir, "generation_{}".format(gen_idx), "species_{}".format(species_idx), self.activity_results_filename)


    def get_winners_path(self, gen_idx):
        return os.path.join(self.proj_results_dir, "generation_{}".format(gen_idx), self.winners_filename)
    def get_HOF_path(self, gen_idx):
        return os.path.join(self.proj_results_dir, "generation_{}".format(gen_idx), self.HOF_filename)

class TL_Evolutioner(Evolutioner):
    def __init__(self, project_name, num_generations=10, num_species=1000, time_step = 0.0005):
        super(TL_Evolutioner, self).__init__(project_name, num_generations=num_generations, num_species=num_species, time_step=time_step)
    def _sample_new_configs(self):
        params_initialiser = TL_ParamsInitialiser()
        configs = params_initialiser.sample_new_configs()
        return configs
    def _calc_fitness_score(self, activity):
        measurer = TL_FitnessMeasurer(activity)
        measurer.build_score_criteria()
        fitness_score = measurer.calc_fitness()
        return fitness_score
    def _initialise_experimenter(self, num_neurons, anatomy_labels):
        exper = TL_Experimenter(num_neurons, anatomy_labels)
        return exper