import pandas as pd

from neuroevolutioner.Evolution import DA_Evolutioner


num_generations = 20
num_species = 1000
time_step = 0.0001
evo = DA_Evolutioner("delayed_activation", num_generations, num_species, time_step)

for i in range(num_generations):
    evo.proliferate_one_generation(i)