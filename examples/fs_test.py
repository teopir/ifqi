from __future__ import print_function
from ifqi.envs import SyntheticToyFS
from ifqi.evaluation import evaluation

mdp = SyntheticToyFS()
dataset = evaluation.collect_episodes(mdp, n_episodes=20)


