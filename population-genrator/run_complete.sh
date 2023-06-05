# Script for population genration
rm population.dat ;
rm -r __pycache__ ;
python genrating.py

#script for weighting 
rm -r __pycache__ ;
rm -r weighted_events ;
mkdir weighted_events;
python weighting.py --dat population.dat --vt vt_m1_m2.hdf5 --n-out 100

#script for scaling
rm -r scaled_events; mkdir scaled_events ; python scaling.py

#moving the events in the desired directory
