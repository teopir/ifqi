import gym
from gym import spaces, envs
import argparse
import numpy as np
import itertools
import time


def gymnumpy_data(state, action=None, reward=None, nextstate=None, done=None):
    """
    Return a list: [state, action, reward, done]
    :param state:
    :param action:
    :param reward:
    :param done:
    :return:
    """
    if action is None \
            and reward is None \
            and nextstate is None and done is None:
        assert isinstance(state, np.ndarray)
        return state.tolist()

    assert isinstance(state, np.ndarray)
    assert isinstance(nextstate, np.ndarray)
    assert isinstance(action, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)

    int_done = 1 if done else 0
    return state.tolist(), action.tolist(), [reward], nextstate.tolist(), [int_done]


class GymCollector:

    def __init__(self, domain):

        supported_domains = {'Pendulum-v0': gymnumpy_data}
        assert domain in supported_domains, 'Unsupported domain {}'.format(domain)

        self.domain = domain
        self.data_processor = supported_domains[domain]

    def collect(self, max_steps = 200, nbepisodes = 100,
                mode="random",
                render=False, fps=None):
        env = envs.make(self.domain)
        ac_space = env.action_space

        fps = fps or env.metadata.get('video.frames_per_second') or 100
        if max_steps == 0: max_steps = env.spec.timestep_limit

        ep = 0
        data = []
        while ep < nbepisodes:
            ep += 1
            print("Starting a new trajectory")
            trajectory = []
            stp1, atp1, rtp1, dtp1 = None, None, None, None

            st = env.reset()
            if render:
                env.render(mode='human')

            for t in range(max_steps) if max_steps else itertools.count():
                done = False
                if isinstance(mode, str):
                    if mode == "random":
                        at = ac_space.sample()
                        stp1, rt, done, info = env.step(at)
                        if render:
                            time.sleep(1.0/fps)
                    elif mode == "human":
                        at = input("type action from {0,...,%i} and press enter: "%(ac_space.n-1))
                        try:
                            at = int(at)
                        except ValueError:
                            print("WARNING: ignoring illegal action '{}'.".format(at))
                            at = 0
                        if at >= ac_space.n:
                            print("WARNING: ignoring illegal action {}.".format(at))
                            at = 0
                        stp1, rt, done, info = env.step(at)
                else:
                    assert hasattr(mode, 'predict')
                    ast = np.array(self.data_processor(st))
                    at = mode.predict(ast)
                    stp1, rt, done, info = env.step(at)

                state, action, reward,  nextstate, absorbing = self.data_processor(st, at, rt, stp1, done)
                trajectory.append(state + action + reward + nextstate + absorbing)

                #next state is current state (t incremented)
                st = stp1

                if render:
                    env.render()

                if done:
                    break


            data.append(trajectory)


            state_dim = len(state)
            action_dim = len(action)
            reward_dim = len(reward)

            print("Done after {} steps".format(t+1))

            if render:
                input("Press enter to continue")

        return data, state_dim, action_dim, reward_dim