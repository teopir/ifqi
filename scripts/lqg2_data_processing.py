import glob
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import time
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

plot = True
mytime = time.time()


ite_knn=101
ite_comp=301
_filter = np.arange(0, ite_comp, (ite_comp-1)/10)
_filter = np.arange(0,201,20)

gh_paths = glob.glob('data/lqg/lqg2_gradients_hessians_*.npy')
#grbf_paths = glob.glob('data/lqg/lqg_gbrf_knn_%s_*.npy' % sigma)
comp_paths = glob.glob('data/lqg/lqg2_comparision_*.npy')

common = set(map(lambda x: x.split('_')[-1], gh_paths)) & \
         set(map(lambda x: x.split('_')[-1], comp_paths)) #& \
         #set(map(lambda x: x.split('_')[-1], grbf_paths))
gh_paths = filter(lambda x: x.split('_')[-1] in common, gh_paths)[:40]
#grbf_paths = filter(lambda x: x.split('_')[-1] in common, grbf_paths)
comp_paths = filter(lambda x: x.split('_')[-1] in common, comp_paths)[:40]

print(len(common))

n = len(gh_paths)
confidence = 0.90

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
    #axes[1].boxplot(np.hstack(gh_arrays[:, 2]).T.squeeze(), labels=names)
    #axes[1].set_title('Hessians')

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

data = [saveme[0], saveme[1].squeeze(), saveme[3].squeeze(), saveme[4], saveme[6]]
#df = pd.DataFrame(data, columns=['Basis function', 'Gradient-mean', 'Gradient-ci', 'Hessian-mean', 'Hessian-ci'])
#df.to_csv('data/csv/lqg_grad_hessian.csv', index_label='Iterations')



comp_labels = comp_arrays[0, 0]
n_comp = len(comp_labels)

comp_mean = np.mean(comp_arrays[:, 1]).squeeze()
comp_std = (np.var(comp_arrays[:, 1]) ** 0.5).squeeze()
#comp_ci = st.t.interval(confidence, n-1, loc=comp_mean, \
#                            scale=comp_std/np.sqrt(n-1))



return_mean = comp_mean[:,:,1].astype(np.float)
return_std = comp_std[:,:,1].astype(np.float)
return_ci = st.t.interval(confidence, n-1, loc=return_mean, \
                            scale=return_std/np.sqrt(n-1))

param_mean = comp_mean[:,:,0]
param_std = comp_std[:,:,0]


if plot:

    _range = np.arange(ite_comp)
    fig, ax = plt.subplots()
    ax.set_xlabel('parameter')
    ax.set_ylabel('iterations')
    fig.suptitle('REINFORCE - Return')

    for i in range(comp_mean.shape[0]):
        y = return_mean[i]
        y_upper = return_ci[1][i] - y
        y_lower = y - return_ci[0][i]
        ax.errorbar(_range[_filter], y[_filter],
                    yerr=[y_lower[_filter], y_upper[_filter]],
                    marker='+', label=comp_labels[i])

    ax.legend(loc='upper right')

'''
    res = None
    _range = np.arange(ite_comp)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    fig.suptitle('REINFORCE - Parameters')
    ax.set_ylabel('k1')
    ax.set_xlabel('iterations')
    ax.set_zlabel('k2')
    ax.plot([0, ite_comp], [-0.6180, -0.6180], [-0.6180, -0.6180], color='k',
            label='Optimal parameter')
    for i in range(comp_mean.shape[0]):
        k1 = [np.asscalar(param_mean[i][j][0]) for j in range(ite_comp)]
        if res is None:
            res = k1
        else:
            res = np.vstack([res, k1])
        k2 = [np.asscalar(param_mean[i][j][1]) for j in range(ite_comp)]
        res = np.vstack([res, k2])
        ax.plot(_range, k1,k2,
                marker=None, label=comp_labels[i])
    ax.legend(loc='upper right')
    res = res.T
    np.savetxt('data/csv/lqg2_param.csv', res, delimiter=';')

    import numpy.linalg as la

    fig = plt.figure()
    k = np.array([-0.6180, -0.6180])
    x = y = np.arange(-3.0, 3.0, 0.05)
    x += k[0]
    y += k[1]
    X, Y = np.meshgrid(x, y)
    for i in range(comp_mean.shape[0]):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        J = -51.
        print(np.trace(hessian_mean[2*i:2*i+2]))
        print(la.eigh(hessian_mean[2*i:2*i+2])[0])
        zs = np.array([J + np.dot(k - np.array([x, y]),
                                  gradient_mean[i].squeeze()) + la.multi_dot(
            [k - np.array([x, y]), hessian_mean[2*i:2*i+2],
             np.transpose(k - np.array([x, y]))]) for x, y in
                       zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)

        ax.plot_surface(X, Y, Z)

        ax.set_zlim(-10 ** 4.5, 100)
        ax.set_xlabel('k1')
        ax.set_ylabel('k2')
        ax.set_zlabel('almost J')
        ax.set_title(names[i])


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
'''