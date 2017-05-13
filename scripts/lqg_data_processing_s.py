import glob
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import time
import pandas as pd

plot = True
mytime = time.time()

sigma = 0.1
n_episodes=5
ite_knn=101
ite_comp=601
_filter = np.arange(0, ite_comp, (ite_comp-1)/10)

gh_paths = glob.glob('data/lqg/lqg_gradients_hessians_%s_%s_*.npy' % (sigma, n_episodes))
#grbf_paths = glob.glob('data/lqg/lqg_gbrf_knn_%s_*.npy' % sigma)
comp_paths = glob.glob('data/lqg/lqg_comparision_%s_%s_*.npy' % (sigma, n_episodes))

common = set(map(lambda x: x.split('_')[-1], gh_paths)) & \
         set(map(lambda x: x.split('_')[-1], comp_paths)) #& \
         #set(map(lambda x: x.split('_')[-1], grbf_paths))
gh_paths = filter(lambda x: x.split('_')[-1] in common, gh_paths)[:40]
#grbf_paths = filter(lambda x: x.split('_')[-1] in common, grbf_paths)
comp_paths = filter(lambda x: x.split('_')[-1] in common, comp_paths)[:40]

print(len(common))

n = len(gh_paths)
confidence = 0.95

gh_arrays = np.array(map(np.load, gh_paths))
#grbf_arrays = np.array(map(np.load, grbf_paths))
comp_arrays = np.array(map(np.load, comp_paths))


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
#np.save('data/confidence_intervals/ci_lqg_gradients_hessians_%s' % mytime, saveme)

data = np.vstack([saveme[0], saveme[1].squeeze(), saveme[3].squeeze(), saveme[4].squeeze(), saveme[6].squeeze()]).T
df = pd.DataFrame(data, columns=['Basis function', 'Gradient-mean', 'Gradient-ci', 'Hessian-mean', 'Hessian-ci'])
df.to_csv('data/csv/lqg_grad_hessian_%s.csv' % sigma, index_label='Iterations')
#------------------------------------------------------------------------------
'''
labels = grbf_arrays[0, 0]
n_knn = len(labels)
print(n_knn)
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
df = df.iloc[np.arange(0, ite_knn, 10)]
df.to_csv('data/csv/lqg_gbrf_knn_%s.csv' % sigma, index_label='Iterations')
'''
#------------------------------------------------------------------------------
comp_labels = comp_arrays[0, 0]
n_comp = len(comp_labels)
comp_mean = np.mean(comp_arrays[:, 1]).astype(np.float64)
comp_std = (np.var(comp_arrays[:, 1]) ** 0.5).astype(np.float64)
comp_ci = st.t.interval(confidence, n-1, loc=comp_mean, \
                            scale=comp_std/np.sqrt(n-1))


if plot:
    _range = np.arange(ite_comp)
    fig, ax = plt.subplots()
    ax.set_xlabel('parameter')
    ax.set_ylabel('iterations')
    fig.suptitle('REINFORCE - Parameter - sigma^2 = %s' % sigma)

    for i in range(comp_mean.shape[0]):
        y = comp_mean[i, :, 0]
        y_upper = comp_ci[1][i, :, 0] - y
        y_lower = y - comp_ci[0][i, :, 0]
        ax.errorbar(_range[_filter], y[_filter],
                    yerr=[y_lower[_filter], y_upper[_filter]],
                    marker='+', label=comp_labels[i])

    ax.legend(loc='upper right')

    _range = np.arange(ite_comp)
    fig, ax = plt.subplots()
    ax.set_xlabel('parameter')
    ax.set_ylabel('iterations')
    fig.suptitle('REINFORCE - Return - sigma^2 = %s' % sigma)

    for i in range(comp_mean.shape[0]):
        y = comp_mean[i, :, 1]
        y_upper = comp_ci[1][i, :, 1] - y
        y_lower = y - comp_ci[0][i, :, 1]
        ax.errorbar(_range[_filter], y[_filter],
                    yerr=[y_lower[_filter], y_upper[_filter]],
                    marker='+', label=comp_labels[i])

    ax.legend(loc='upper right')

saveme = np.zeros(4, dtype=object)
saveme[0] = comp_labels
saveme[1] = comp_mean
saveme[2] = comp_std
saveme[3] = comp_mean - comp_ci[0]
#np.save('data/confidence_intervals/ci_lqg_comparision_%s' % mytime, saveme)

second = ['Parameter', 'Return', 'Gradient']
third = ['Mean', 'Std', 'Error']

data = np.stack(saveme[1:], axis=2).transpose([0, 3, 2, 1]).reshape(9*n_comp, ite_comp).T
col_names1 = np.zeros((n_comp, 3, 3), dtype=object)
col_names2 = np.zeros((n_comp, 3, 3), dtype=object)
col_names3 = np.zeros((n_comp, 3, 3), dtype=object)

for i in range(len(comp_labels)):
    col_names1[i, :, :] = comp_labels[i]
for i in range(len(second)):
    col_names2[:, i, :] = second[i]
for i in range(len(third)):
    col_names3[:, :, i] = third[i]

col_names = (col_names1 + '---' + col_names2 + '---' + col_names3).reshape(9*n_comp, 1).ravel()
df = pd.DataFrame(data, columns=col_names)
df = df.iloc[_filter]
df.to_csv('data/csv/lqg_comparision_%s.csv' % sigma, index_label='Iterations')