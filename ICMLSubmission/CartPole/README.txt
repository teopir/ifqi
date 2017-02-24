#Run the experiment -same for all experiments
python ../../examples/experimentThreadManager.py CartPoleContET CartPoleContET.json NThread 

#Plot
python ../../plot3D.py CartPoleContET score -1 -1

python ../../plot3D.py CartPoleContMLP score -1 -1 -m

python ../../plot3D.py CartPoleContMLPDS score --size -m -l 

python ../../plot.py CartPoleContETDS score --size

