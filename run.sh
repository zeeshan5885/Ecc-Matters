time python3 -mcProfile new_powerlaw_mcmc.py ecc_events/*.dat vt_m1_m2.hdf5 myout.hdf5 --n-walkers 20 --n-samples 500;
mv myout.hdf5 ./plotting/
