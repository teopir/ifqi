iFQI
******

**iFQI is a toolkit for solving Reinforcement Learning problem using bot Fitted Q-Iteration or Boosted Fitted Q-Iteration **

.. contents:: **Contents of this document**
   :depth: 2

Installation
============

You can perform a minimal install of ``ifqi`` with:

.. code:: shell

	git clone https://github.com/teopir/ifqi.git
	git checkout tosatto
	cd ifqi
	pip install -e .

Installing everything
---------------------

To install the whole set of features, you will need additional packages installed.
You can install everything by running ``pip install -e '.[all]'``.


What's new
----------
- 2016-XX-YY: Initial release

How to set and run and experiment
=================================

Prepare a configuration file
----------------------------

First of all you need a json file where its described how the experiment will be performed. When you'll run the experiment with a same json configuration file, you'll get exactly the same outcome.
In the configuration file you'll define for examples

1) The environemnt 
2) Which regressors will you use (you can define many of them in the same experiment, with different parameters)
3) You'll define the number of dataset that you'll use to learn. The datasets will be generated from your selected environment, with random policy.
4) You can choose the dataset sizes: the size correspond to the number of episodes that will be used to compose your datasets. Remember that if you will insert more than one size, in your experiment you will generate n dataset with size1, n dataset with size2, and so on.. 
5) You can choose the number of repetitions: if the repetition is equal to 1, than one regressor will perform the learning procedure only once, but the fitting of a regressor is often stochastic, so here there is the possibility to run the learning procedure over the same dataset different number of times and collect all the outcomes
6) You can choose how often you would like to evaluate your learning procedure (so how many iteration you vould like to run the policy found and collect the scores). You can also choose how many episodes run every times to evaluate the policy found: if the environment is deterministic, you will set this number to one, but will be a good idea to evaluate your policy more than once if your environment is stochastic
7) The number of FQI iterations (this will depend more or less in how many iteration you belive you'll find a good policy.) If you have a lot of iterations (like 100), it is wise to don't evaluate your policy every iteration, but something like once every 5 iteration for example. The evaluation of a policy is an expensive procedure.
8) ...

Run the experiment
------------------

All you have to do is just prepare the json file as described above, and then call

.. code:: shell

	python examples/experimentThreadManager.py folderName configFilename.json threadNumber --cont
	
The **folderName** this will be even the name of the folder where will be saved the outcome of the experiment. Will be a good idea to give the same name of the json file to avoid confusion. 
The **configFilename.json** file is the name of the configuration file as described above.
As ExperimentThreadManager (as the name suggest) can run a different number of thread to exploit thread level parallelism, you can set **threadNumber** 1 if you don't want to have parallelism, or more than one if you want to use more than one thread (ideally you will set this parameter equal to the number of cores you would like to use).
There is also the possibility to continue one experiment after have killed it with the option **-c** or **-cont**: Of course the thread that were killed will need to be executed, but the ones already executed and terminated will be skipped (if they will have saved the last iteration of the algorithm on disk).


ICML Submission: Boosted Fitted Q-Iteration
-------------------------------------------

It is possible, after the correct installation of the present library, to execute all the experiment presented in the paper, and so to let the reader to verify by its own the result shown, but also to start from these experiment and run a self-made ones.
The goal of available experiments is - as stated in the paper - to compare FQI and B-FQI with different environments and under different aspects.

All the configuration files, followed by a brief description will be found in the folder **ICML Submission**.


