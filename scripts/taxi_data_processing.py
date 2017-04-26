import glob
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import time
import pandas as pd

plot = True
mytime = time.time()

gh_paths = glob.glob('data/taxi/taxi_gradients_hessians_*.npy')
comp_paths = glob.glob('data/taxi/taxi_comparision_*.npy')

n = len(gh_paths)
confidence = 0.95

gh_arrays = np.array(map(np.load, gh_paths))
comp_arrays = np.array(map(np.load, comp_paths))

names = gh_arrays[0, 0]
#Compute gradient and hessian statistics
gh_mean, gh_std, gh_error = [], [], []
for i in range(1, gh_arrays.shape[1]):
    gh_mean.append(np.mean(gh_arrays[:, i]))
    gh_std.append(np.std(gh_arrays[:, i]))
    ci = st.t.interval(confidence, n-1, loc=gh_mean[-1], \
                            scale=gh_std[-1]/np.sqrt(n-1))
    gh_error.append(gh_mean[-1] - ci[0])


if plot:
    _range = np.arange(234)
    fig, axes = plt.subplots()
    for i in range(len(names)):
        y = gh_mean[5][i]
        y_upper = gh_error[5][i]
        y_lower = gh_error[5][i]
        axes.errorbar(_range, gh_mean[5][i], yerr=[y_upper, y_lower], label =names[i])
    #axes.legend(loc='upper right')
    plt.yscale('symlog', linthreshy=1e-12)

    fig, axes = plt.subplots(nrows=4, ncols=1)
    axes[0].boxplot(np.vstack(gh_arrays[:, 3]), labels=names)
    axes[0].set_title('Gradients norm 2')
    axes[1].boxplot(np.vstack(gh_arrays[:, 4]), labels=names)
    axes[1].set_title('Gradients norm inf')
    axes[2].boxplot(np.vstack(gh_arrays[:, 5]), labels=names)
    axes[2].set_title('Max eigenvalue')
    axes[3].boxplot(np.vstack(gh_arrays[:, 7]), labels=names)
    axes[3].set_title('Traces')


res = np.stack([gh_mean[5], gh_std[5], gh_error[5]]).transpose([1, 0, 2]).reshape(27, 234).T
col_names = map(str.__add__, np.repeat(names, 3), np.tile(['-Mean', '-Std', '-Error'], 9))
df = pd.DataFrame(res, columns=col_names)
df.to_csv('data/csv/taxi_eigenvalues.csv', index_label='index')

#------------------------------------------------------------------------------
comp_labels = comp_arrays[0, 0]
comp_mean = np.mean(comp_arrays[:, 1])
comp_std = (np.var(comp_arrays[:, 1]) ** 0.5)


return_mean = comp_mean[:, :, 1].astype(np.float64)
return_std = comp_std[:, :, 1].astype(np.float64)
return_ci = st.t.interval(confidence, n-1, loc=return_mean, \
                            scale=return_std/np.sqrt(n-1))
return_error =  return_mean - return_ci[0]

if plot:
    _range = np.arange(201)
    fig, ax = plt.subplots()
    ax.set_xlabel('return')
    ax.set_ylabel('iterations')
    fig.suptitle('REINFORCE - Return')

    for i in range(comp_mean.shape[0]):
        y = return_mean[i, :]
        y_upper = return_error[i, :]
        y_lower = return_error[i, :]
        ax.errorbar(_range, y,
                    yerr=[y_lower, y_upper],
                    marker='+', label=comp_labels[i])

    ax.legend(loc='upper right')


titles = map(lambda x: x+'-mean', comp_labels) + map(lambda x: x+'-error', comp_labels)
res = np.vstack([return_mean, return_error]).T[range(0, 201, 10)]
df = pd.DataFrame(res, columns=titles, index=range(0, 201, 10))
df.to_csv('data/csv/taxi_return.csv', index_label='Iterations')

#-------------------------------------------------------------------------------
from policy import BoltzmannPolicy
from ifqi.evaluation import evaluation
from ifqi.envs import TaxiEnv
import copy
state_features = np.load('taxi_features_weights/taxi_state_features.npy')
action_weights = np.load('taxi_features_weights/taxi_action_weights.npy')
expert_policy =  BoltzmannPolicy(state_features, action_weights)
policy = copy.deepcopy(expert_policy)

mdp = TaxiEnv()
mdp.horizon = 100

comp_d_kl_mean = np.zeros((comp_arrays[:, 1][0].shape[0], 201))
comp_d_kl_std = np.zeros((comp_arrays[:, 1][0].shape[0], 201))

for i in range(comp_arrays[:, 1][0].shape[0]): #for every basis function
    for j in range(0, 201, 10): #for every iteration (not all!)
        policy.set_parameter(comp_mean[i, j, 0], build_gradient_hessian=False)
        d_kl = []
        print('%s-%s' % (i, j))
        for k in range(100):
            dataset = evaluation.collect_episode(mdp, policy)
            states = dataset[:, 0].astype(np.int)
            actions = dataset[:, 1].astype(np.int)
            #compute the kl
            p_expert = expert_policy.pi[states, actions]
            p_hat = policy.pi[states, actions]
            kl = (p_expert * np.log(p_expert / (p_hat + 1e-24) + 1e-24)).sum() / len(dataset)
            d_kl.append(kl)

        comp_d_kl_mean[i, j] = np.mean(d_kl)
        comp_d_kl_std[i, j] = np.std(d_kl)

comp_d_kl_mean = comp_d_kl_mean[:, range(0, 201, 10)]
comp_d_kl_std = comp_d_kl_std[:, range(0, 201, 10)]
comp_d_kl_ci = st.t.interval(confidence, 100-1, loc=comp_d_kl_mean, \
                            scale=comp_d_kl_std/np.sqrt(100-1))
comp_d_kl_error = comp_d_kl_mean - comp_d_kl_ci[0]

if plot:
    _range = np.arange(21)
    fig, ax = plt.subplots()
    ax.set_xlabel('kl div')
    ax.set_ylabel('iterations')
    fig.suptitle('REINFORCE - kl div')

    for i in range(comp_mean.shape[0]):
        y = comp_d_kl_mean[i, :]
        y_upper = comp_d_kl_error[i, :]
        y_lower = comp_d_kl_error[i, :]
        ax.errorbar(_range, y,
                    yerr=[y_lower, y_upper],
                    marker='+', label=comp_labels[i])

    ax.legend(loc='upper right')

titles = map(lambda x: x+'-mean', comp_labels) + map(lambda x: x+'-error', comp_labels)
res = np.vstack([comp_d_kl_mean, comp_d_kl_error]).T
df = pd.DataFrame(res, columns=titles, index=range(0, 201, 10))
df.to_csv('data/csv/taxi_kl.csv', index_label='Iterations')

