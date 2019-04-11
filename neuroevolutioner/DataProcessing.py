
import numpy as np
from scipy.signal import convolve2d


def calculate_firing_rate(firing_activity, kernel_size, time_step):
    """
    Args:
        firing_activity: (np.darray) with shape (time, num_neurons).
        kernel_size: (float) The width of sliding window (in second) for rate calculation, along the time axis
        time_step: (float) Simulation time step
    """
    kernel_size_steps = int(kernel_size/time_step)
    # mean_kernel = np.


