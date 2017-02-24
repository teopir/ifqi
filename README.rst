iFQI
******

**iFQI is a toolkit for solving Reinforcement Learning problem using bot Fitted Q-Iteration or Boosted Fitted Q-Iteration**

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

ICML Submission: Boosted Fitted Q-Iteration
===========================================

How to run the experiments
--------------------------

It is possible, after the correct installation of the present library, to execute all the experiment presented in the paper, and so to let the reader to verify by its own the result shown, but also to start from these experiment and run a self-made ones.
The goal of available experiments is - as stated in the paper - to compare FQI and B-FQI with different environments and under different aspects.

All the configuration files, followed by a brief description will be found in the folder **ICML Submission**.
Place yourself in one of the folders contained in **ICMLSubmission** (depending on which experiment you would like to run)

.. code:: shell

	python examples/experimentThreadManager.py folderName configFilename.json threadNumber --cont
	
The **folderName** this will be even the name of the folder where will be saved the outcome of the experiment. Will be a good idea to give the same name of the json file to avoid confusion. 
The **configFilename.json** file is the name of the configuration file as described above.
As ExperimentThreadManager (as the name suggest) can run a different number of thread to exploit thread level parallelism, you can set **threadNumber** 1 if you don't want to have parallelism, or more than one if you want to use more than one thread (ideally you will set this parameter equal to the number of cores you would like to use).
There is also the possibility to continue one experiment after have killed it with the option **-c** or **-cont**: Of course the thread that were killed will need to be executed, but the ones already executed and terminated will be skipped (if they will have saved the last iteration of the algorithm on disk).

Plots the results
-----------------

Once an experiment has ended, you probably would like to plot it. All the variable subject of study are saved on the disk in many different numpy files. The easiest way to plot the results in a human-readable way is to use **examples/plot.py** or **examples/plot3D.py** depending on the situation.

The plots have many parameters that were used to provide nice plots, in order to don't confuse the reader, we provide here the right line to plot each different experiment.

Plotting the bicycle
--------------------

These are the plots of the plots of the performance of the bicycle in the article and in appendix

.. code:: shell
	python ../../plot.py BicycleBalET score 
	python ../../plot.py BicycleBalMLP score  -m
	python ../../plot.py BicycleBalET steps 
	python ../../plot.py BicycleBalMLP steps  -m
	
Plotting the CartPole
---------------------

Dataset Analysis

.. code:: shell

	python ../../plot.py CartPoleContMLPDS score --size -m -l 
	python ../../plot.py CartPoleContETDS score --size -l
	
Complexity Analysis
	
.. code:: shell	
	python ../../plot3D.py CartPoleContET score -1 -1 -t
	python ../../plot3D.py CartPoleContMLP score -1 -1 -m -t

Plotting the SwingUpPendulum
----------------------------

Dataset Analysis

.. code:: shell

	python ../../plot.py SwingUpPendulumMLPDS score --size -m -l 
	python ../../plot.py SwingUpPendulumETDS score --size -l
	
Dataset Analysis
	
.. code:: shell	

	python ../../plot3D.py SwingUpPendulumET score -1 -1
	python ../../plot3D.py SwingUpPendulumMLP score -1 -1 -m

use parameter -tl when the legend cover the lines.







