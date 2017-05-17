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
m=40
confidence = 0.95
#_filter = range(0, 151, 15)
_filter = [1000]

expert = {0.0 : (8.813, 0.03815), 0.01 : (8.425, 0.04475), 0.05 : (6.743, 0.06871), 0.1 : (4.397, 0.06242)}

gh_paths = glob.glob('data/taxi/taxi_gradients_hessians_%s_%s_*.npy' % (epsilon, n_episodes))
comp_paths = glob.glob('data/taxi/taxi_comparision_%s_%s_*.npy' % (epsilon, n_episodes))
lpal_paths = glob.glob('data/taxi/lpal_%s_%s_*.npy' % (epsilon, n_episodes))
bc_paths = glob.glob('data/taxi/bc_%s_%s_*.npy' % (epsilon, n_episodes))

common = set(map(lambda x: x.split('_')[-1], gh_paths)) & \
         set(map(lambda x: x.split('_')[-1], comp_paths)) & \
         set(map(lambda x: x.split('_')[-1], lpal_paths))
gh_paths = filter(lambda x: x.split('_')[-1] in common, gh_paths)[:m]
lpal_paths = filter(lambda x: x.split('_')[-1] in common, lpal_paths)[:m]
comp_paths = filter(lambda x: x.split('_')[-1] in common, comp_paths)[:m]

n = len(gh_paths)
print(n)

gh_arrays = np.array(map(np.load, gh_paths))
comp_arrays = np.array(map(np.load, comp_paths))
lpal_arrays = np.array(map(np.load, lpal_paths))
bc_arrays =  np.array(map(np.load, bc_paths))

names = gh_arrays[0, 0]
#Compute gradient and hessian statistics
gh_mean, gh_std, gh_error = [], [], []
for i in range(1, gh_arrays.shape[1]):
    gh_mean.append(np.mean(gh_arrays[:, i]))
    gh_std.append(np.std(gh_arrays[:, i]))
    ci = st.t.interval(confidence, n-1, loc=gh_mean[-1], \
                            scale=gh_std[-1]/np.sqrt(n-1))
    gh_error.append(gh_mean[-1] - ci[0])

plot = False
if plot:
    _range = np.arange(240)
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

plot = True
res = np.stack([gh_mean[5], gh_std[5], gh_error[5]]).transpose([1, 0, 2]).reshape(3*n_features, 240).T
col_names = map(str.__add__, np.repeat(names, 3), np.tile(['-Mean', '-Std', '-Error'], n_features))
df = pd.DataFrame(res, columns=col_names)
df.to_csv('data/csv/taxi_eigenvalues_%s.csv' % epsilon, index_label='index')

#------------------------------------------------------------------------------
comp_labels = comp_arrays[0, 0]
comp_mean = np.mean(comp_arrays[:, 1])
comp_std = (np.var(comp_arrays[:, 1]) ** 0.5)

lpal_mean = np.mean(map(np.array, lpal_arrays[:, 1]), axis=0)
lpal_std = (np.var(map(np.array, lpal_arrays[:, 1]), axis=0) ** 0.5)
lpal_ci = st.t.interval(confidence, n-1, loc=lpal_mean, \
                            scale=lpal_std/np.sqrt(n-1))
lpal_error = lpal_mean - lpal_ci[0]


bc_mean = np.mean(bc_arrays[:, 1])
bc_std = (np.var(bc_arrays[:, 1]) ** 0.5)
bc_ci = st.t.interval(confidence, n-1, loc=bc_mean, \
                            scale=bc_std/np.sqrt(n-1))
bc_error = bc_mean - bc_ci[0]



return_mean = comp_mean[:, :, 1].astype(np.float64)
return_std = comp_std[:, :, 1].astype(np.float64)
return_ci = st.t.interval(confidence, n-1, loc=return_mean, \
                            scale=return_std/np.sqrt(n-1))
return_error =  return_mean - return_ci[0]

return_mean = np.vstack([return_mean,
                         np.tile(lpal_mean[:, np.newaxis], (1,iterations)),
                         np.tile(bc_mean, (1,iterations)),
                         np.tile(expert[epsilon][0], (1, iterations))])
return_error = np.vstack([return_error,
                          np.tile(lpal_error[:, np.newaxis], (1,iterations)),
                          np.tile(bc_error, (1,iterations)),
                          np.tile(expert[epsilon][1], (1, iterations))])
comp_labels = comp_labels + ['LPAL natural features', 'LPAL ECO-R model free', 'LPAL ECO-R model based', 'BC', 'Expert']

if plot:
    _range = np.arange(iterations)
    fig, ax = plt.subplots()
    ax.set_ylabel('return')
    ax.set_xlabel('iterations')
    fig.suptitle('REINFORCE - Return - epsilon=%s' % epsilon)

    print(len(return_mean[0, :]))
    print(len(_range))

    for i in [0,1,2,3,10,11]:
        y = return_mean[i, :]
        y_upper = return_error[i, :]
        y_lower = return_error[i, :]
        ax.errorbar(_filter, y[_filter],
                    yerr=[y_lower[_filter], y_upper[_filter]],
                    marker='+', label=comp_labels[i])

    ax.legend(loc='lower right')

    fig, ax = plt.subplots()
    ax.set_ylabel('return')
    ax.set_xlabel('iterations')
    fig.suptitle('REINFORCE - Return - epsilon=%s' % epsilon)
    for i in [2,3,4,5,6,7,8,9]:
        y = return_mean[i, :]
        y_upper = return_error[i, :]
        y_lower = return_error[i, :]
        ax.errorbar(_filter, y[_filter],
                    yerr=[y_lower[_filter], y_upper[_filter]],
                    marker='+', label=comp_labels[i])

    ax.legend(loc='lower right')


titles = map(lambda x: x+'-mean', comp_labels) + map(lambda x: x+'-error', comp_labels)
res = np.vstack([return_mean, return_error]).T[_filter]
df = pd.DataFrame(res, columns=titles, index=_filter)
df.to_csv('data/csv/taxi_return_%s.csv' % epsilon, index_label='Iterations')


#-------------------------------------------------------------------------------
from policy import BoltzmannPolicy, EpsilonGreedyBoltzmannPolicy, TabularPolicy
from ifqi.evaluation import evaluation
from ifqi.envs import TaxiEnv
import copy
state_features = np.load('taxi_features_weights/taxi_state_features.npy')
action_weights = np.load('taxi_features_weights/taxi_action_weights.npy')
expert_policy =  EpsilonGreedyBoltzmannPolicy(epsilon, state_features, action_weights)
policy = copy.deepcopy(expert_policy)

mdp = TaxiEnv()
mdp.horizon = 100

comp_d_kl_mean = np.zeros((comp_arrays[:, 1][0].shape[0]+4, iterations))
comp_d_kl_std = np.zeros((comp_arrays[:, 1][0].shape[0]+4, iterations))

for i in range(comp_arrays[:, 1][0].shape[0]): #for every basis function
    print(i)
    for j in _filter: #for every iteration (not all!)
        print('\t%s' %j)
        d_kl = []
        for l in range(n):
            policy.set_parameter(np.array(comp_arrays[l, 1])[i, j, 0], build_gradient=False,
                                 build_hessian=False)
            d_kl_trial = []
            for k in range(100):
                dataset = evaluation.collect_episode(mdp, expert_policy)
                states = dataset[:, 0].astype(np.int)
                actions = dataset[:, 1].astype(np.int)
                # compute the kl
                p_expert = expert_policy.pi[states, actions]
                p_hat = policy.pi[states, actions]
                kl = (p_expert * np.log(
                    p_expert / (p_hat + 1e-24) + 1e-24)).sum() / len(dataset)
                d_kl_trial.append(kl)
            d_kl.append(np.mean(d_kl_trial))

        comp_d_kl_mean[i, j] = np.mean(d_kl)
        comp_d_kl_std[i, j] = np.std(d_kl)


for i in range(comp_arrays[:, 1][0].shape[0], comp_arrays[:, 1][0].shape[0]+3):
    d_kl = []
    for l in range(n):
        policy = TabularPolicy(lpal_arrays[l, 0][i-comp_arrays[:, 1][0].shape[0]])
        d_kl_trial = []
        for k in range(100):
            dataset = evaluation.collect_episode(mdp, expert_policy)
            states = dataset[:, 0].astype(np.int)
            actions = dataset[:, 1].astype(np.int)
            # compute the kl
            p_expert = expert_policy.pi[states, actions]
            p_hat = policy.pi[states, actions]
            kl = (p_expert * np.log(
                p_expert / (p_hat + 1e-24) + 1e-24)).sum() / len(dataset)
            d_kl_trial.append(kl)
        d_kl.append(np.mean(d_kl_trial))
    comp_d_kl_mean[i, :] = np.mean(d_kl)
    comp_d_kl_std[i, :] = np.std(d_kl)

d_kl = []
i = comp_arrays[:, 1][0].shape[0] + 3
for l in range(n):
    policy = TabularPolicy(bc_arrays[l, 0])
    d_kl_trial = []
    for k in range(100):
        dataset = evaluation.collect_episode(mdp, expert_policy)
        states = dataset[:, 0].astype(np.int)
        actions = dataset[:, 1].astype(np.int)
        # compute the kl
        p_expert = expert_policy.pi[states, actions]
        p_hat = policy.pi[states, actions]
        kl = (p_expert * np.log(
            p_expert / (p_hat + 1e-24) + 1e-24)).sum() / len(dataset)
        d_kl_trial.append(kl)
    d_kl.append(np.mean(d_kl_trial))
comp_d_kl_mean[i, :] = np.mean(d_kl)
comp_d_kl_std[i, :] = np.std(d_kl)

comp_d_kl_mean = comp_d_kl_mean[:, _filter]
comp_d_kl_std = comp_d_kl_std[:, _filter]
comp_d_kl_ci = st.t.interval(confidence, n-1, loc=comp_d_kl_mean, \
                            scale=comp_d_kl_std/np.sqrt(n-1))
comp_d_kl_error = comp_d_kl_mean - comp_d_kl_ci[0]



if plot:



    _range = np.arange(iterations)
    fig, ax = plt.subplots()
    ax.set_ylabel('KL-div')
    ax.set_xlabel('iterations')
    fig.suptitle('REINFORCE - KL-div - epsilon=%s' % epsilon)


    for i in [0,1,2,3,10]:
        y = comp_d_kl_mean[i, :]
        y_upper = comp_d_kl_error[i, :]
        y_lower = comp_d_kl_error[i, :]
        ax.errorbar(_filter, y,
                    yerr=[y_lower, y_upper],
                    marker='+', label=comp_labels[i])

    ax.legend(loc='upper right')

    fig, ax = plt.subplots()
    ax.set_ylabel('KL-div')
    ax.set_xlabel('iterations')
    fig.suptitle('REINFORCE - KL-div - epsilon=%s' % epsilon)
    for i in [2,3,4,5,6,7,8,9]:
        y = comp_d_kl_mean[i, :]
        y_upper = comp_d_kl_error[i, :]
        y_lower = comp_d_kl_error[i, :]
        ax.errorbar(_filter, y,
                    yerr=[y_lower, y_upper],
                    marker='+', label=comp_labels[i])

    ax.legend(loc='upper right')

titles = map(lambda x: x+'-mean', comp_labels[:-1]) + map(lambda x: x+'-error', comp_labels[:-1])
res = np.vstack([comp_d_kl_mean, comp_d_kl_error]).T
df = pd.DataFrame(res, columns=titles, index=_filter)
df.to_csv('data/csv/taxi_kl_%s.csv' % epsilon, index_label='Iterations')

