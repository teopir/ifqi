#Run the experiment -same for all experiments
python ../../examples/experimentThreadManager.py SwingUpPendulumET SwingUpPendulumET.json NThread 

#Plot
python ../../plot3D.py SwingUpPendulumET score -1 -1

python ../../plot3D.py SwingUpPendulumMLP score -1 -1 -m

python ../../plot3D.py SwingUpPendulumMLPDS score --size -m -l 

python ../../plot.py SwingUpPendulumETDS score --size

