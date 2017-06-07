import glob
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import time
import pandas as pd

plot = True
mytime = time.time()

epsilon=0.1
n_episodes = 100
iterations = 1001
n_features = 7
m=1000
confidence = 0.80
_filter = range(0, 301, 30)

expert = {0.0 : (8.813, 0.03815), 0.01 : (8.425, 0.04475), 0.05 : (6.743, 0.06871), 0.1 : (4.397, 0.06242)}

comp_paths = glob.glob('data/taxi/natural/taxi_comparision_%s_%s_*.npy' % (epsilon, n_episodes))
lpal_paths = glob.glob('data/taxi/natural/lpal_%s_%s_*.npy' % (epsilon, n_episodes))
bc_paths = glob.glob('data/taxi/bc_%s_%s_*.npy' % (epsilon, n_episodes))

common = set(map(lambda x: x.split('_')[-1], comp_paths)) & \
         set(map(lambda x: x.split('_')[-1], lpal_paths))
lpal_paths = filter(lambda x: x.split('_')[-1] in common, lpal_paths)[:m]
comp_paths = filter(lambda x: x.split('_')[-1] in common, comp_paths)[:m]

n=len(comp_paths)
print(n)

comp_arrays = np.array(map(np.load, comp_paths))
lpal_arrays = np.array(map(np.load, lpal_paths))
bc_arrays =  np.array(map(np.load, bc_paths))

#------------------------------------------------------------------------------
comp_labels = comp_arrays[0, 0]
comp_mean = np.mean(comp_arrays[:, 1])
comp_std = (np.var(comp_arrays[:, 1]) ** 0.5)

lpal_mean = np.mean(map(np.array, lpal_arrays[:, 1]), axis=0)
lpal_std = (np.var(map(np.array, lpal_arrays[:, 1]), axis=0) ** 0.5)
lpal_ci = st.t.interval(confidence, n-1, loc=lpal_mean, \
                            scale=lpal_std/np.sqrt(n-1))
lpal_error = lpal_mean - lpal_ci[0]




return_mean = comp_mean[:, :, 1].astype(np.float64)
return_std = comp_std[:, :, 1].astype(np.float64)
return_ci = st.t.interval(confidence, n-1, loc=return_mean, \
                            scale=return_std/np.sqrt(n-1))
return_error =  return_mean - return_ci[0]

return_mean = np.vstack([return_mean,
                         np.tile(lpal_mean[:, np.newaxis], (1,iterations)),
                         np.tile(expert[epsilon][0], (1, iterations))])
return_error = np.vstack([return_error,
                          np.tile(lpal_error[:, np.newaxis], (1,iterations)),
                          np.tile(expert[epsilon][1], (1, iterations))])
comp_labels = comp_labels + ['LPAL natural features']

if plot:

    fig, ax = plt.subplots()
    ax.set_ylabel('return')
    ax.set_xlabel('iterations')
    fig.suptitle('REINFORCE - Return - epsilon=%s' % epsilon)
    for i in [0,1]:
        y = return_mean[i, :]
        y_upper = return_error[i, :]
        y_lower = return_error[i, :]
        ax.errorbar(_filter, y[_filter],
                    yerr=[y_lower[_filter], y_upper[_filter]],
                    marker='+', label=comp_labels[i])

    ax.legend(loc='lower right')

comp_labels = ['ME-PVF', 'HESS-PVF', 'LPAL-PVF']
titles = map(lambda x: x+'-mean', comp_labels) + map(lambda x: x+'-error', comp_labels)
res = np.vstack([return_mean[:3], return_error[:3]]).T[_filter]
df = pd.DataFrame(res, columns=titles, index=_filter)
df.to_csv('data/csv/taxi_return_pvf_%s.csv' % epsilon, index_label='Iterations')