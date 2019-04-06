import pandas as pd
import os

base_dir = "experiment_results/time_learning/generation_0"

hof_path = os.path.join(base_dir, "hall_of_fame.csv")
winner_path = os.path.join(base_dir, "winners.csv")

hof = pd.read_csv(hof_path)
winners = pd.read_csv(winner_path)

hof["gen_idx"] = 0
winners["gen_idx"] = 0

print(hof.head())
print(winners.head())