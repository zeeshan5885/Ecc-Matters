# Script for population genration
rm population.dat ;
rm scaled_population.dat
rm weighted_population.dat
rm -r __pycache__ ;
python3 genrating.py

#script for weighting 
rm -r __pycache__ ;
rm -r weighted_events ;
mkdir weighted_events;
python3 weighting.py --dat population.dat --vt vt_m1_m2.hdf5 --n-out 10

#Adding Gaussian Uncertainities in the injections

rm -r rPE_events;
mkdir rPE_events;
python3 real_PE.py;

#script for scaling
rm -r scaled_events; mkdir scaled_events ; python3 scaling.py;


# making scatter plots
python3 plotting.py;

#copying generated data to the appropriate direcoty to run the inference
cp -r weighted_events/*.dat ../weighted_events;
cp -r scaled_events/*.dat ../original_pop_code/scaled_events;