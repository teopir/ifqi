import glob
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import time
import pandas as pd

confidence=0.9
n=10
iet=20

expert_paths = glob.glob('data/ch/expert_returns_*.npy' )[:n]
eco_return = glob.glob('data/ch/eco_returns_*.npy' )[:n]
eco_explo_return = glob.glob('data/ch/eco_returns_explo2_*.npy' )

expert_arrays = np.array(map(np.load, expert_paths)) / 20
eco_return_arrays = np.array(map(np.load, eco_return)) / 20
eco_explo_return_arrays = np.array(map(np.load, eco_explo_return)) / 20

eco_explo_return_arrays = eco_explo_return_arrays[eco_explo_return_arrays[:,-1].argsort()[::-1][:n]]

print(expert_arrays.max())

expert_mean = np.mean(expert_arrays, axis=0)
expert_std = np.std(expert_arrays, axis=0)
expert_ci = st.t.interval(confidence, n-1, loc=expert_mean, \
                            scale=expert_std/np.sqrt(n-1))

eco_mean = np.mean(eco_return_arrays, axis=0)
eco_std = np.std(eco_return_arrays, axis=0)
eco_ci = st.t.interval(confidence, n-1, loc=eco_mean, \
                            scale=eco_std/np.sqrt(n-1))

eco_explo_mean = np.mean(eco_explo_return_arrays, axis=0)
eco_explo_std = np.std(eco_explo_return_arrays, axis=0)
eco_explo_ci = st.t.interval(confidence, n-1, loc=eco_explo_mean, \
                            scale=eco_explo_std/np.sqrt(n-1))


_range = np.arange(1, 21)
fig, ax = plt.subplots()
ax.set_xlabel('FQI iteration')
ax.set_ylabel('average return')
fig.suptitle('FQI - Average return')

ax.errorbar(_range, expert_mean,
            yerr=abs(expert_ci-expert_mean),
            marker='+', label='Reward function', color='r')

ax.errorbar(_range, eco_mean,
            yerr=abs(eco_ci-eco_mean),
            marker='+', label='ECO-R epsilon=0.0', color='b')

ax.errorbar(_range, eco_explo_mean,
            yerr=abs(eco_explo_ci-eco_explo_mean),
            marker='+', label='ECO-R epsilon=0.2', color='g')

ax.legend(loc='upper right')


res = np.vstack([eco_explo_mean, abs(eco_explo_ci-eco_explo_mean)])
np.savetxt('ch.csv',res.T, delimiter=';')