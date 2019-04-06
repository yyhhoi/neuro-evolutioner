import pandas as pd

from neuroevolutioner.Evolution import TL_Evolutioner

gens = 0
evo = TL_Evolutioner("time_learning", gens, 1000)


evo.proliferate_one_generation(gens)