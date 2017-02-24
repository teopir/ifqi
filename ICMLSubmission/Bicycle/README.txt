#Run the experiment -same for all experiments
python ../../examples/experimentThreadManager.py BicycleBalET BicycleBalET.json NThread 

#Plot
python ../../plot.py BicycleBalET score 

python ../../plot.py BicycleBalMLP score  -m


