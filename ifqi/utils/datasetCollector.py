import numpy as np
from ifqi.experiment.Experiment import getMDP


def collect(mdp):
    sast, r = mdp.runEpisode(None, True, False)[3, 4]
    r = np.array(r)

    sarst = np.concatenate((np.concatenate((sast[:, 0:mdp.state_dim], r.T),
                                           axis=1),
                            sast[:, -1 - mdp.state_dim]

if __name__ == '__main__':
    mdpName = 'CarOnHill'
    nEpisodes = 100

    mdp = getMDP(mdpName)
    sarst = collect(mdp)

    sarst.save('../dataset/' + mdpName)
