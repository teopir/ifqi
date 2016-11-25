from __future__ import print_function
import ifqi.envs as env
from ifqi.envs.utils import get_space_info
from ifqi.evaluation import evaluation
from ifqi.algorithms.selection import RFS, IFS
from sklearn.ensemble import ExtraTreesRegressor

mdp = env.SyntheticToyFS()
state_dim, action_dim, reward_dim = get_space_info(mdp)
nextstate_idx = state_dim + action_dim
reward_idx = nextstate_idx + state_dim

dataset = evaluation.collect_episodes(mdp, n_episodes=20)

selector = IFS(estimator=ExtraTreesRegressor(n_estimators=100),
               scale=True,
               verbose=1)
fs = RFS(feature_selector=selector,
         features_names=['S1', 'S2', 'S3', 'A'],
         verbose=1)

state=dataset[:, 0:state_dim]
actions=dataset[:, state_dim:nextstate_idx]
next_states=dataset[:, nextstate_idx:reward_idx]
reward=dataset[:, reward_idx]

print(dataset[:10, :])

# fs.fit(state, actions, next_states, reward)

