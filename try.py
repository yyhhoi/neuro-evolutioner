import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


activity_shape = (12001, 10)
time_step = 0.0005
duration_size = int(0.5/time_step)
step_size = int(0.25/time_step)
rest = int(3/time_step)
# # One oblique line
# activity1 = np.zeros(activity_shape)
# for i in range(10):
#     activity1[0 + int(i * step_size) :duration_size + int(i * step_size), i] = 1

# # Two oblique lines
# activity2 = np.zeros(activity_shape)
# for i in range(10):
#     activity2[0 + int(i * step_size) :duration_size + int(i * step_size), i] = 1
#     activity2[rest + int(i * step_size) : rest+duration_size + int(i * step_size), i] = 1

# # Whole region
# activity3 = np.zeros(activity_shape)
# activity3[int(1/time_step) : int(5/time_step), :] = 1

# # Striaght line
# activity4 = np.zeros(activity_shape)
# activity4[int(3/time_step) : int(3.5/time_step), :] = 1


# fig, ax = plt.subplots(2,2, figsize=(8,12))
# fig.suptitle("Input pattern")
# ax[0,0].imshow(activity1.T, aspect="auto")
# ax[0,0].set_title("A1")
# ax[0,1].imshow(activity2.T, aspect="auto")
# ax[0,1].set_title("A2")
# ax[1,0].imshow(activity3.T, aspect="auto")
# ax[1,0].set_title("A3")
# ax[1,1].imshow(activity4.T, aspect="auto")
# ax[1,1].set_title("A4")
# ax = ax.ravel()
# for ax_each in ax:
#     ax_each.set_xlim(0, activity_shape[0])
#     ax_each.set_ylim(0, activity_shape[1]-1)



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

    train_S_period1 = activity_pattern[0:tl_accu_times["train_S"], 0]
    train_S_period2 = activity_pattern[tl_accu_times["rest1"]:tl_accu_times["test_S"], 0]
    train_S_plus = np.mean(np.append(train_S_period1, train_S_period2))

    train_S_non_period1 = activity_pattern[tl_accu_times["train_S"]:tl_accu_times["rest1"], 0]
    train_S_non_period2 = activity_pattern[tl_accu_times["test_S"]:, 0]
    train_S_minus = np.mean(np.append(train_S_non_period1, train_S_non_period2)) * -1

    train_A_period1 = activity_pattern[tl_accu_times["train_ISI"]:tl_accu_times["train_A"], 9]
    train_A_period2 = activity_pattern[tl_accu_times["test_ISI"]:tl_accu_times["Test_A"], 9]
    train_A_plus = np.mean(np.append(train_A_period1, train_A_period2))

    train_A_non_period1 = activity_pattern[0:tl_accu_times["train_ISI"], 9]
    train_A_non_period2 = activity_pattern[tl_accu_times["train_A"]:tl_accu_times["test_ISI"], 9]
    train_A_minus = np.mean(np.append(train_A_non_period1, train_A_non_period2)) * -1

    rest_non_period = activity_pattern[tl_accu_times["train_A"]:tl_accu_times["rest1"], 1:9]
    rest_minus = np.mean(rest_non_period)*2

    score = train_S_plus + train_S_minus + train_A_plus + train_A_minus + rest_minus

    return score

absolute_pattern, convolution_kernal, absolute_normalisation, convolution_kernal_template = gen_fitness_filter()

fig2, ax2 = plt.subplots(2, figsize=(8,12))
fig2.suptitle("Filters")
ax2[0].imshow(absolute_pattern.T, aspect="auto")
ax2[0].set_title("Abs")
ax2[1].imshow(convolution_kernal.T, aspect="auto")
ax2[1].set_title("Conv1")
ax2 = ax2.ravel()
for ax_each2 in ax2:
    ax_each2.set_xlim(0, activity_shape[0])
    ax_each2.set_ylim(0, activity_shape[1]-1)

# one_stripe_conv1 = convolve2d(activity1, convolution_kernal, mode="valid")
# two_stripes_conv1 = convolve2d(activity2, convolution_kernal, mode="valid")
# whole_conv1 = convolve2d(activity3, convolution_kernal, mode="valid")
# line_conv1 = convolve2d(activity4, convolution_kernal, mode="valid")

# one_stripe_abs = (activity1 * absolute_pattern).sum()/absolute_normalisation
# two_stripes_abs = (activity2 * absolute_pattern).sum()/absolute_normalisation
# whole_abs = (activity3 * absolute_pattern).sum()/absolute_normalisation
# line_abs = (activity4 * absolute_pattern).sum()/absolute_normalisation


# one_stripe_conv1 = one_stripe_conv1/one_stripe_conv1.shape[0]
# two_stripes_conv1 = two_stripes_conv1/two_stripes_conv1.shape[0]
# whole_conv1 = whole_conv1/whole_conv1.shape[0]
# line_conv1 = line_conv1/line_conv1.shape[0]

# print(activity1.shape)
# print(convolution_kernal.shape)

# def fitness(arr):
    
#     score = np.sum(arr>0)/arr.squeeze().shape[0]
#     return score

# fig3, ax3 = plt.subplots(2,2, figsize=(8,12))
# fig3.suptitle("Conv1")
# ax3[0,0].plot(one_stripe_conv1.squeeze())
# ax3[0,0].set_title("one_stripe: Fittness: %0.4f\nAbs: %0.4f" % (fitness(one_stripe_conv1), one_stripe_abs))
# ax3[0,1].plot(two_stripes_conv1.squeeze())
# ax3[0,1].set_title("two_stripes:Fittness: %0.4f\nAbs: %0.4f" % (fitness(two_stripes_conv1), two_stripes_abs))
# ax3[1,0].plot(whole_conv1.squeeze())
# ax3[1,0].set_title("whole:Fittness: %0.4f\nAbs: %0.4f" % (fitness(whole_conv1), whole_abs))
# ax3[1,1].plot(line_conv1.squeeze())
# ax3[1,1].set_title("line:Fittness: %0.4f\nAbs: %0.4f" % (fitness(line_conv1), line_abs))


plt.show()



