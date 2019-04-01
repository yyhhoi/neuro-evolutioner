from neuroevolutioner.Environments import Simulation
from neuroevolutioner.Ensembles import Ensemble_AdEx
from neuroevolutioner.ParamsGenomes import get_SMC_configs
import matplotlib.pyplot as plt
import numpy as np


simenv = Simulation(5, epsilon=0.001)
num_neurons, configs = get_SMC_configs()
ensemble = Ensemble_AdEx(simenv, num_neurons, configs)
ensemble.initialize_parameters(configs)
while simenv.sim_stop == False:
    print(simenv.getTime())
    ensemble.state_update()
    simenv.increment()
