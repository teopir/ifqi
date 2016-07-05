from __future__ import print_function

#sys.path.append(os.path.abspath('../'))

import numpy as np
from ifqi.models.incr import WideRegressor
from ifqi.models.sum_regressor import SumRegressor


seed = 0 
np.random.seed(seed)
wideRegr = WideRegressor(n_input=3, n_output=1, hidden_neurons=[50]*20, 
              n_h_layer_beginning=2,optimizer='rmsprop', act_function=["relu"]*20)
np.random.seed(seed)
sumRegr = SumRegressor(n_input=3, n_output=1, hidden_neurons=[50]*20, 
              n_h_layer_beginning=2,optimizer='rmsprop', act_function=["relu"]*20)

dataset = np.random.rand(50,3)
answer = np.random.rand(50,1)
query = np.random.rand(50,3)

fit_params = {'nb_epoch':300, 'batch_size':100, 'verbose':0}

for i in range(20):
    seed+=1
    np.random.seed(seed)
    wideRegr.fit(dataset,answer.ravel(),**fit_params)
    answer_1 = wideRegr.predict(query)
    np.random.seed(seed)
    sumRegr.fit(dataset,answer.ravel(), **fit_params)
    answer_2 = sumRegr.predict(query)

    assert np.array_equal(answer_1,answer_2),"Test failed"
    print ("Test Passed")
    
    seed+=1
    np.random.seed(seed)
    wideRegr.adapt(i)
    np.random.seed(seed)
    sumRegr.adapt(i)

