import pandas as pd

from neuroevolutioner.Evolution import TL_Evolutioner


num_generations = 5
evo = TL_Evolutioner("time_learning", num_generations, 1000)
for i in range(num_generations):
    evo.proliferate_one_generation(i)