import glob
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import time
import pandas as pd

plot = True
mytime = time.time()

sigma = 1.
ite_knn=301
ite_comp=601
_filter = np.arange(0, ite_comp, (ite_comp-1)/10)
_filter_knn = np.arange(0, ite_knn, (ite_knn-1)/10)
#gh_paths = glob.glob('data/lqg/lqg_gradients_hessians_%s_*.npy' % sigma)
grbf_paths = glob.glob('data/lqg/lqg_gbrf_knn_%s_*.npy' % sigma)[:40]
#comp_paths = glob.glob('data/lqg/lqg_comparision_%s_*.npy' % sigma)

print(len(grbf_paths))
n = len(grbf_paths)
confidence = 0.95

grbf_arrays = np.array(map(np.load, grbf_paths))

#------------------------------------------------------------------------------

labels = grbf_arrays[0, 0]
n_knn = len(labels)

iter_mean = np.mean(grbf_arrays[:, 1]).astype(np.float64)
iter_std = (np.var(grbf_arrays[:, 1]) ** 0.5).astype(np.float64)
iter_ci = st.t.interval(confidence, n-1, loc=iter_mean, \
                            scale=iter_std/np.sqrt(n-1))

if plot:
    _range = np.arange(ite_knn)
    fig, ax = plt.subplots()
    ax.set_xlabel('parameter')
    ax.set_ylabel('iterations')
    fig.suptitle('REINFORCE - Parameter')

    for i in range(iter_mean.shape[0]):
        y = iter_mean[i, :, 0]
        y_upper = iter_ci[1][i, :, 0] - y
        y_lower = y - iter_ci[0][i, :, 0]
        ax.errorbar(_range, y,
                    yerr=[y_lower, y_upper],
                    marker='+', label=labels[i])

    ax.legend(loc='upper right')

saveme = np.zeros(4, dtype=object)
saveme[0] = labels
saveme[1] = iter_mean
saveme[2] = iter_std
saveme[3] = iter_mean - iter_ci[0]
#np.save('data/confidence_intervals/ci_lqg_gbrf_knn_%s' % mytime, saveme)

second = ['Parameter', 'Return', 'Gradient']
third = ['Mean', 'Std', 'Error']

data = np.stack(saveme[1:], axis=2).transpose([0, 3, 2, 1]).reshape(9 * n_knn, ite_knn).T
col_names1 = np.zeros((n_knn, 3, 3), dtype=object)
col_names2 = np.zeros((n_knn, 3, 3), dtype=object)
col_names3 = np.zeros((n_knn, 3, 3), dtype=object)

for i in range(len(labels)):
    col_names1[i, :, :] = labels[i]
for i in range(len(second)):
    col_names2[:, i, :] = second[i]
for i in range(len(third)):
    col_names3[:, :, i] = third[i]

col_names = (col_names1 + '---' + col_names2 + '---' + col_names3).reshape(9 * n_knn, 1).ravel()
df = pd.DataFrame(data, columns=col_names)
df = df.iloc[_filter_knn]
df.to_csv('data/csv/lqg_gbrf_knn_%s.csv' % sigma, index_label='Iterations')