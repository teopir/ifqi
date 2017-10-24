from ifqi.envs.lqg1d import LQG1D
from ifqi.utils.policy import GaussianPolicy
from ifqi.evaluation import evaluation
import numpy as np

mdp = LQG1D()
k = mdp.computeOptimalK()

state_features = lambda state: state

policy = GaussianPolicy(n_dimensions=1,
                        n_parameters=1,
                        covariance_matrix=0.1,
                        parameters=k,
                        state_features=state_features)

trajectories = evaluation.collect_episodes(mdp, policy, n_episodes=10)

state_index = np.arange(0, np.prod(mdp.observation_space.shape))
action_index = state_index[-1] + np.arange(0, np.prod(mdp.action_space.shape))
reward_index = [action_index[-1] + 1]
next_state_index = reward_index[-1] + np.arange(0, np.prod(mdp.observation_space.shape))
absorbing_flag_index = [next_state_index[-1] + 1]
end_episode_flag_index = [absorbing_flag_index[-1] + 1]

return_mean, return_std, steps_mean, steps_std  = evaluation.evaluate_policy(mdp, policy, n_episodes=10)



gradients = policy.gradient_log(trajectories[:, state_index], trajectories[:, action_index])