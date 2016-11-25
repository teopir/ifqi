from __future__ import print_function
import ifqi.envs as env
from ifqi.envs.utils import get_space_info
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_dataset
from ifqi.algorithms.selection import RFS, IFS
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np

# np.random.seed(3452)

mdp = env.SyntheticToyFS()
state_dim, action_dim, reward_dim = get_space_info(mdp)
nextstate_idx = state_dim + action_dim + reward_dim
reward_idx = action_dim + state_dim

# dataset: s, a, r, s'
dataset = evaluation.collect_episodes(mdp, n_episodes=50)
check_dataset(dataset, state_dim, action_dim, reward_dim)

selector = IFS(estimator=ExtraTreesRegressor(n_estimators=50),
               scale=True, verbose=1)
fs = RFS(feature_selector=selector,
         features_names=np.array(['S0', 'S1', 'S2', 'S3', 'A0', 'A1']),
         verbose=1)

state, actions, reward, next_states = \
    split_dataset(dataset, state_dim, action_dim, reward_dim)

# print(dataset[:10, :])

fs.fit(state, actions, next_states, reward)
print(fs.get_support())  # this are the selected features, it should be [s0,s2, a0]
