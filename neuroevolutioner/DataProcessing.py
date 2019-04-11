
import numpy as np
from scipy.signal import convolve
import pandas as pd
import os


def calculate_firing_rate(firing_activity, kernel_size, time_step):
    """
    Args:
        firing_activity: (np.darray) with shape (time, num_neurons).
        kernel_size: (float) The width of sliding window (in second) for rate calculation, along the time axis
        time_step: (float) Simulation time step
    """
    num_neurons = firing_activity.shape[1]
    kernel_size_steps = int(kernel_size/time_step)
    mean_kernel = np.ones((kernel_size_steps, 1))
    output = convolve(firing_activity, mean_kernel, mode="same") / kernel_size
    return output

class DA_data_processing():
    def __init__(self, project_name, vis_dir, time_step = 0.0001):
        self.project_name = project_name
        self.vis_dir = vis_dir
        self.time_step = time_step
        pass
    
    def produce_firing_rate(self, gen_idx, species_idx,  kernel_size = 50e-3, df_col = [i + 2 for i in range(10)]):
        results_dir = os.path.join(self.vis_dir, self.project_name, "gen-{}_species-{}".format(gen_idx, species_idx))
        activity_path = os.path.join(results_dir, "activity.csv")
        firing_rate_save_path = os.path.join(results_dir , "firing_rate.csv")
        df_act = pd.read_csv(activity_path)
        act_np = np.array(df_act.iloc[:, df_col])
        firing_rate = calculate_firing_rate(act_np, kernel_size, self.time_step)
        df_rate = df_act.copy()
        df_rate.iloc[:, df_col] = firing_rate
        df_rate.to_csv(firing_rate_save_path, index=False)



if __name__ == "__main__":
    pass


