import pandas as pd

from neuroevolutioner.Evolution import TL_Evolutioner


num_generations = 5
evo = TL_Evolutioner("time_learning", num_generations, 5, time_step=0.0005)
# evo.evaluate_one_generation(0)
# evo.select_winners(0, 0.5)
for i in range(num_generations):
    evo.proliferate_one_generation(i)