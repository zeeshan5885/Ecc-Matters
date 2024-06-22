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
python3 weighting.py --dat population.dat --vt vt_1_200_1000.hdf5 --n-out 100

#Adding Gaussian Uncertainities in the injections

rm -r ecc_events;
mkdir ecc_events;
python3 fake_PE.py;

#script for scaling
rm -r scaled_events; mkdir scaled_events ; python3 scaling.py;


# making scatter plots
python3 plotting.py;

#copying generated data to the appropriate direcoty to run the inference
cp ecc_events/*.dat ../ecc_events;
cp scaled_events/*.dat ../original_pop_code/scaled_events;
