time python3 -mcProfile new_powerlaw_mcmc.py weighted_events/*.dat vt_m1_m2.hdf5 myout.hdf5 --n-walkers 10 --n-samples 400;
mv myout.hdf5 ./plotting/
