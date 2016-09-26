import unittest
from ifqi.envs.acrobot import Acrobot
from ifqi.envs.carOnHill import CarOnHill
from ifqi.envs.cartPole import CartPole
from ifqi.envs.invertedPendulum import InvPendulum
from ifqi.envs.lqg1d import LQG1D
from ifqi.envs.swingPendulum import SwingPendulum
import ifqi.evaluation.evaluation as evaluate
import numpy as np

"""
Here we test all the environments
"""


class TestEnvironment(unittest.TestCase):
    def test_step(self):

        def checkStep(env):
            """
            Here we test if 
                - env.step don't raise error
                - shape and type of returned values from step are as expected
            """

            lastState = env.reset()
            self.assertTrue(env.observation_space.low.shape == lastState.shape,
                            "State incoherent with the space defined")
            for _ in xrange(20):
                action = env.action_space.sample()
                state, r, done, _ = env.step(action)

                self.assertTrue(lastState.shape == state.shape,
                                "Previous state has different shape by the actual: " + str(lastState) + " vs " + str(
                                    state))
                self.assertTrue(type(r) in set([float, int, long, np.float, np.float16, np.float32, np.float64]),
                                "Reward should be a scalar with type long, or int, or long but " + str(type(r)))
                self.assertTrue(np.isscalar(r), "Reward should be a scalar")
                self.assertTrue(type(done) in set([bool]), "Done should be of boolean type")

                # even if is not necessary
                lastState = state
                if done:
                    break

        checkStep(Acrobot())
        # checkStep(Bicycle())
        checkStep(CarOnHill())
        checkStep(CartPole())
        checkStep(InvPendulum())
        checkStep(LQG1D())
        checkStep(SwingPendulum())

    class RandomPolicy:

        def __init__(self, environment):
            self.environment = environment

        def drawAction(self, state):
            return self.environment.action_space.sample()

    def test_evaluation(self):
        return

        # TODO: fix this
        def checkEvaluation(env):
            """
            Here I fix the seed, and I make sure that the values returned from an evaluation are the same as expected
            """
            rndPolicy = self.RandomPolicy(env)
            env.seed(0)
            J_eval, _ = env.evaluate(rndPolicy)

            env.seed(0)

            done = False
            env.reset()

            J = 0.

            print("eval")
            t = 0
            while not done and t < env.horizon:
                action = env.action_space.sample()
                if t < 10:
                    print(action)
                state, r, done, _ = env.step(action)
                J += env.gamma ** t * r
                t += 1

            print(J_eval, J)
            self.assertTrue(np.isclose(J_eval, J),
                            "Two evaluation should be almost the same, we have instead: " + str(J_eval) + " " + str(J))

        checkEvaluation(Acrobot())
        # checkStep(Bicycle())
        checkEvaluation(CarOnHill())
        checkEvaluation(CartPole())
        checkEvaluation(InvPendulum())
        checkEvaluation(LQG1D())
        checkEvaluation(SwingPendulum())

    def test_CollectEpisode(self):

        def checkCollect(env):
            """check if the type of the data given by CollectEpisode is consistent"""
            data = evaluate.collectEpisode(env)

            stateDim = env.observation_space.shape[0]

            self.assertTrue(data.shape[1] == 3 + stateDim * 2 + 1,
                            "Dataset shape is not consistent with the environment description")
            endEpisode = data[:, 0]
            state = data[:, 1:1 + stateDim]
            action = data[:, stateDim]
            reward = data[:, stateDim + 1]
            nextState = data[:, stateDim + 2: 2 * stateDim + 2]
            absorbing = data[:, -1]

        checkCollect(Acrobot())
        print "CarOnHill"
        # checkStep(Bicycle())
        checkCollect(CarOnHill())
        checkCollect(CartPole())
        checkCollect(InvPendulum())
        checkCollect(LQG1D())
        checkCollect(SwingPendulum())


if __name__ == "__main__":
    unittest.main()
