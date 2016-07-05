"""ExperimentGenerator.py

this script interfaces with ExpMan library, and creates in automated way a number of "experiments"
to run. The regards "PendulumResultsExp.py".

"""





import sys

#https://github.com/SamuelePolimi/ExperimentManager.git
from ExpMan.core.ExperimentManager import ExperimentManager


if len(sys.argv) < 2:
    raise "Number of parameters not sufficient"
    
#example
#python experimentGenerator results/IncrementalModel 2 incr,mlp,boost AF 10 n_neuron=10 n_epoch=20
folder = sys.argv[1]
test = sys.argv[2]
models = sys.argv[3].split(",")                         #models to try
data = sys.argv[4]
n_test = int(sys.argv[5])

params = []
for arg in sys.argv[6:]:
    params.append(arg)
    
man = ExperimentManager()
man.loadExperiment(folder)
expTest = man.openExperimentTest(int(test))

comment = raw_input("Insert a comment: ")
for model in models:
    #addTestCase(self, name, cmd, params, comment)
    #TODO: set program name
    case = expTest.addTestCase(model, "examples.pendulumResultsExp", ["model="+model] +  params, comment)
    
    for dt in data:
        for n in xrange(0,n_test):
            case.addSample(["dataset=" + dt],"nsample="+str(n))
                