from neuroevolutioner.Environments import Simulation
from neuroevolutioner.Ensembles import Ensemble_AdEx
from neuroevolutioner.utils import load_pickle, write_pickle
from neuroevolutioner.ParamsGenomes import get_SMC_configs, convert_config2genes, convert_genes2config, test_if_two_configs_are_equal
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import pickle

record_path = "ensemble_test.csv"
gene_path = "gene_test.pickle"
configs_path = "configs.pickle"
simenv = Simulation(0.01, epsilon=0.001)
configs = get_SMC_configs()
num_neurons = configs["Meta"]["num_neurons"]



I_ext = np.random.uniform(0,1, (num_neurons,)) * 10e-12

ensemble = Ensemble_AdEx(simenv, num_neurons, configs, record_path=record_path)
ensemble.initialize_parameters(configs)
while simenv.sim_stop == False:
    print(simenv.getTime())
    ensemble.I_ext = I_ext
    ensemble.state_update()
    simenv.increment()
    