import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


t = np.linspace(-1,1,100)
t_plus = np.abs(t)


x1 = np.sqrt(np.abs(t))
x1[t < 0] = -np.sqrt(np.abs(t))[t < 0]


fig ,ax = plt.subplots()

ax.plot(t,x1)
ax.plot(t,t)
plt.show()


