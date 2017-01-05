from __future__ import print_function

import ifqi.envs as env
from ifqi.algorithms.lspi import LSPI
from ifqi.envs.utils import get_space_info
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_data_for_fqi
from ifqi.models.linear import Linear
from ifqi.models.regressor import Regressor
from ifqi.preprocessors.features import PolyFeatures

mdp = env.CarOnHill()
state_dim, action_dim, reward_dim = get_space_info(mdp)
nextstate_idx = state_dim + action_dim + reward_dim
reward_idx = action_dim + state_dim

# dataset: s, a, r, s'
dataset = evaluation.collect_episodes(mdp, n_episodes=50)
check_dataset(dataset, state_dim, action_dim, reward_dim)

regressor_params = dict(features=dict(name='poly', params=dict(degree=5)))
regressor = Regressor(Linear, **regressor_params)
lspi = LSPI(regressor, state_dim, action_dim, mdp.action_space.values,
            mdp.gamma)

sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)

lspi.fit(sast, r)

values = evaluation.evaluate_policy(mdp, lspi, initial_states=mdp.initial_states)

print(values)
