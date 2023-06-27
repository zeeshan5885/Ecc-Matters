# Script for population genration
rm population.dat ;
rm scaled_population.dat
rm weighted_population.dat
rm -r __pycache__ ;
python genrating.py

#script for weighting 
rm -r __pycache__ ;
rm -r weighted_events ;
mkdir weighted_events;
python weighting.py --dat population.dat --vt vt_m1_m2.hdf5 --n-out 100

#Adding Gaussian Uncertainities in the injections

rm -r rPE_events;
mkdir rPE_events;
python real_PE.py;

#script for scaling
rm -r scaled_events; mkdir scaled_events ; python scaling.py;


# making scatter plots
python plotting.py
