import glob
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import time
import pandas as pd

plot = True
mytime = time.time()

gh_paths = glob.glob('data/lqg_e/lqg_gradients_hessians_*.npy')
grbf_paths = glob.glob('data/lqg_gbrf_knn_*.npy')
comp_paths = glob.glob('data/lqg_comparision_*.npy')
comp_paths_e = glob.glob('data/lqg_e/lqg_comparision_*.npy')

n = len(gh_paths)
confidence = 0.95

gh_arrays = np.array(map(np.load, gh_paths))
grbf_arrays = np.array(map(np.load, grbf_paths))
comp_arrays = np.array(map(np.load, comp_paths))
comp_arrays_e = np.array(map(np.load, comp_paths_e))

for i in range(80):
    comp_arrays[i, 1] = np.vstack([comp_arrays[i, 1], comp_arrays_e[i, 1]])
    comp_arrays[i, 0].append('Maximum entropy')

names = gh_arrays[0, 0]
#Compute gradient and hessian statistics
gradient_mean = np.mean(gh_arrays[:, 1])
gradient_std = np.std(gh_arrays[:, 1])
hessian_mean = np.mean(gh_arrays[:, 2])
hessian_std = np.std(gh_arrays[:, 2])

#Compute confidence intervals
gradient_ci = st.t.interval(confidence, n-1, loc=gradient_mean, \
                            scale=gradient_std/np.sqrt(n-1))

hessian_ci = st.t.interval(confidence, n-1, loc=hessian_mean, \
                            scale=hessian_std/np.sqrt(n-1))
if plot:
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].boxplot(np.hstack(gh_arrays[:, 1]).T, labels=names)
    axes[0].set_title('Gradients')
    axes[1].boxplot(np.hstack(gh_arrays[:, 2]).T.squeeze(), labels=names)
    axes[1].set_title('Hessians')

#Save data
saveme = np.zeros(7, dtype=object)
saveme[0] = names
saveme[1] = gradient_mean
saveme[2] = gradient_std
saveme[3] = gradient_mean - gradient_ci[0]
saveme[4] = hessian_mean
saveme[5] = hessian_std
saveme[6] = hessian_mean - hessian_ci[0]
np.save('data/confidence_intervals/ci_lqg_gradients_hessians_%s' % mytime, saveme)

#------------------------------------------------------------------------------
labels = grbf_arrays[0, 0]
iter_mean = np.mean(grbf_arrays[:, 1]).astype(np.float64)
iter_std = (np.var(grbf_arrays[:, 1]) ** 0.5).astype(np.float64)
iter_ci = st.t.interval(confidence, n-1, loc=iter_mean, \
                            scale=iter_std/np.sqrt(n-1))

if plot:
    _range = np.arange(101)
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
np.save('data/confidence_intervals/ci_lqg_gbrf_knn_%s' % mytime, saveme)

second = ['Parameter', 'Return', 'Gradient']
third = ['Mean', 'Std', 'Error']

data = np.stack(saveme[1:], axis=2).transpose([0, 3, 2, 1]).reshape(108, 101).T
col_names1 = np.zeros((12, 3, 3), dtype=object)
col_names2 = np.zeros((12, 3, 3), dtype=object)
col_names3 = np.zeros((12, 3, 3), dtype=object)

for i in range(len(labels)):
    col_names1[i, :, :] = labels[i]
for i in range(len(second)):
    col_names2[:, i, :] = second[i]
for i in range(len(third)):
    col_names3[:, :, i] = third[i]

col_names = (col_names1 + '---' + col_names2 + '---' + col_names3).reshape(108, 1).ravel()
df = pd.DataFrame(data, columns=col_names)
df = df.iloc[np.arange(0, 101, 10)]
df.to_csv('data/csv/ci_lqg_gbrf_knn.csv', index_label='Iterations')

#------------------------------------------------------------------------------
comp_labels = comp_arrays[0, 0]
comp_mean = np.mean(comp_arrays[:, 1]).astype(np.float64)
comp_std = (np.var(comp_arrays[:, 1]) ** 0.5).astype(np.float64)
comp_ci = st.t.interval(confidence, n-1, loc=comp_mean, \
                            scale=comp_std/np.sqrt(n-1))

if plot:
    _range = np.arange(401)
    fig, ax = plt.subplots()
    ax.set_xlabel('parameter')
    ax.set_ylabel('iterations')
    fig.suptitle('REINFORCE - Parameter')

    for i in range(comp_mean.shape[0]):
        y = comp_mean[i, :, 0]
        y_upper = comp_ci[1][i, :, 0] - y
        y_lower = y - comp_ci[0][i, :, 0]
        ax.errorbar(_range, y,
                    yerr=[y_lower, y_upper],
                    marker='+', label=comp_labels[i])

    ax.legend(loc='upper right')

saveme = np.zeros(4, dtype=object)
saveme[0] = comp_labels
saveme[1] = comp_mean
saveme[2] = comp_std
saveme[3] = comp_mean - comp_ci[0]
np.save('data/confidence_intervals/ci_lqg_comparision_%s' % mytime, saveme)

second = ['Parameter', 'Return', 'Gradient']
third = ['Mean', 'Std', 'Error']

data = np.stack(saveme[1:], axis=2).transpose([0, 3, 2, 1]).reshape(99, 401).T
col_names1 = np.zeros((11, 3, 3), dtype=object)
col_names2 = np.zeros((11, 3, 3), dtype=object)
col_names3 = np.zeros((11, 3, 3), dtype=object)

for i in range(len(comp_labels)):
    col_names1[i, :, :] = comp_labels[i]
for i in range(len(second)):
    col_names2[:, i, :] = second[i]
for i in range(len(third)):
    col_names3[:, :, i] = third[i]

col_names = (col_names1 + '---' + col_names2 + '---' + col_names3).reshape(99, 1).ravel()
df = pd.DataFrame(data, columns=col_names)
df = df.iloc[np.arange(0, 401, 40)]
df.to_csv('data/csv/ci_lqg_comparision.csv', index_label='Iterations')

