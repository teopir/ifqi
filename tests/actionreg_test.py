from __future__ import  print_function
import numpy as np

from ifqi.models.actionregressor import ActionRegressor

class TestRegressor(object):

    def fit(self, X, y, **kwargs):
        self.p = np.mean(X, axis=0)
        self.n = X.shape[0]

    def predict(self, X, **kwargs):
        return self.p[0]

def test_function(discrete_actions, D, C):
    print(discrete_actions)


    print('-'*20)
    print(discrete_actions)
    ar = ActionRegressor(TestRegressor(), discrete_actions, 1e-5)
    print('Actions in AR: {}'.format(ar._actions))

    # check few properties (shape and that all the actions are preserved)
    assert len(ar._actions.shape) == 2
    assert ar._actions.shape[0] == len(discrete_actions)
    for el in discrete_actions:
        found = False
        for row in ar._actions:
            if np.allclose(row, el):
                found = True
                break
        assert found

    print()
    print('D: {}'.format(D))

    ar.fit(D, D)


    for i in range(len(ar._models)):
        el = ar._models[i]
        print(el.p)
        print(el.n)
        assert el.n == C[i]
        assert np.allclose(el.p, ar._actions[i])
        print('.'*10)

    X = ar.predict(D)
    assert np.allclose(X, D[:,0])

if __name__ == "__main__":
    np.random.seed(513506739)
    discrete_actions = [[4.5,3], [4.5655,3.66], [4.4666, 0.000001]]
    discrete_actions = np.linspace(-1,1.3134, 5)
    idxs = np.random.randint(0, len(discrete_actions), 10)
    D = np.array([discrete_actions[el] for el in idxs]).reshape(-1, len(discrete_actions[0]) if isinstance(discrete_actions[0], list) else 1)
    C = [1, 4, 1, 2, 2]
    test_function(discrete_actions, D, C)


    print('\n###STARTING NEW TEST')
    discrete_actions = [[4.5,3], [4.5655,3.66], [4.4666, 0.000001]]
    idxs = np.random.randint(0, len(discrete_actions), 10)
    D = np.array([discrete_actions[el] for el in idxs]).reshape(-1, len(discrete_actions[0]) if isinstance(discrete_actions[0], list) else 1)
    C = [2, 1, 7]
    test_function(discrete_actions, D, C)