import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

dummy_df = np.zeros((12001, 10))
step_size = 400
for i in range(10):
    dummy_df[i*step_size: (i+1)*step_size, i] = 1

dummy_df = np.zeros((12001, 10))
step_size = 400
dummy_df[2500: 2500+400, :] = 1
# for i in range(10):
#     dummy_df[i*step_size + 1000: (i+1)*step_size + 1000, i] = 1
tl_accu_times = {
    "train_S": 1,
    "train_ISI": 1.5,
    "train_A": 2,
    "rest1": 4,
    "test_S": 5,
    "test_ISI": 5.5,
    "Test_A":6,
    "repeat": 1
}


def gen_fitness_filter(time_step=0.0005, tl_accu_times=tl_accu_times):
    absolute_pattern = np.zeros((12001, 10))
    # normalised to 1s = 1 mark
    absolute_pattern[0:int(tl_accu_times["train_S"]/time_step), 0] = 1 # Train_S
    overlap_start = (tl_accu_times["train_A"] + tl_accu_times["train_ISI"])/2
    absolute_pattern[int(overlap_start/time_step):int(tl_accu_times["train_A"]/time_step), 9 ] = 1/0.25 # Train_A

    absolute_pattern[int(tl_accu_times["rest1"]/time_step):int(tl_accu_times["test_S"]/time_step), 0] = 1 # test_S
    overlap_start = (tl_accu_times["Test_A"] + tl_accu_times["test_ISI"])/2
    absolute_pattern[int(overlap_start/time_step):int(tl_accu_times["Test_A"]/time_step), 9] = 1/0.25 # test_A

    absolute_normalisation = np.sum(absolute_pattern[absolute_pattern > 0])

    convolution_kernal = np.ones((int(tl_accu_times["train_A"]/time_step), 10))*-1
    convolution_kernal[int((tl_accu_times["train_S"]/4)/time_step):int(tl_accu_times["train_S"]/time_step), 0] = 1
    overlap_start = (tl_accu_times["train_ISI"] + tl_accu_times["train_A"])/2
    convolution_kernal[int(overlap_start/time_step):int(tl_accu_times["train_A"]/time_step), 9] = 1
    for i in range(0, 8):
        idx = i + 1
        convolution_kernal[int((0.5 + i/7)/time_step):int((0.75 + idx/7)/time_step) , idx] = 1
    
    return absolute_pattern, convolution_kernal, absolute_normalisation

absolute_pattern, convolution_kernal, absolute_normalisation = gen_fitness_filter()


dummy_df[0:convolution_kernal.shape[0], : ]  = np.clip(convolution_kernal, 0, 1)

convolved = convolve2d(dummy_df, convolution_kernal, mode="valid")
fitness_score = np.max(convolved/convolved.shape[0])
absolute_score = 1
# absolute_score = (absolute_pattern * dummy_df)).sum()/absolute_normalisation

print(convolved.shape)
print(absolute_normalisation)
fig, ax = plt.subplots(2,2)
ax = ax.ravel()
ax[0].imshow(absolute_pattern, aspect="auto")
ax[0].set_ylim(0, 12001)
ax[0].set_xlim(9, 0)
ax[1].imshow(convolution_kernal, aspect="auto")
ax[1].set_ylim(0, convolution_kernal.shape[0])
ax[1].set_xlim(9, 0)

ax[2].imshow(dummy_df, aspect="auto")
ax[2].set_ylim(0, dummy_df.shape[0])
ax[2].set_xlim(9, 0)

ax[3].plot(convolved.squeeze())

fig.suptitle("Convolve = %0.4f\n Absolute = %0.4f"%(fitness_score, absolute_score))
plt.show()





