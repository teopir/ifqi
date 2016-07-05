"""blackboxGenerator.py

this script interfaces with ExpMan library, and creates in automated way a number of "experiments"
to run. The regards "blackboxResults.py".

"""

import sys

#https://github.com/SamuelePolimi/ExperimentManager.git
from ExpMan.core.ExperimentManager import ExperimentManager


if len(sys.argv) < 6:
    raise "Number of parameters not sufficient"
    
#example
#python experimentGenerator results/blackbox 0 extra linearmodel,randomgen 1 n_estimators=50
folder = sys.argv[1]
test = sys.argv[2]
models = sys.argv[3].split(",")                         #models to try
data = sys.argv[4].split(",")
n_test = int(sys.argv[5])

params = []
#for arg in sys.argv[6:]:
#    params.append(arg)
    
man = ExperimentManager()
man.loadExperiment(folder)
expTest = man.openExperimentTest(int(test))

comment = raw_input("Insert a comment: ")
for model in models:
    #addTestCase(self, name, cmd, params, comment)
    #TODO: set program name
    case = expTest.addTestCase(model, "examples.blackboxResults", ["model="+model] , comment)
    #case = expTest.addTestCase(model, "examples.blackboxResults", ["model="+model] +  params, comment)
    for dt in data:
        for n in xrange(0,n_test):
            case.addSample(["dataset=" + dt],"nsample="+str(n))
                
