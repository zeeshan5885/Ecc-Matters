#! /bin/bash
time python3 -mcProfile original_powerlaw_mcmc.py scaled_events/*.dat vt_m1_m2.hdf5 myout_org.hdf5 --n-walkers 10 --n-samples 50

# Check the exit status of python run
if [ $? -eq 0 ]; then
    mv myout_org.hdf5 ../plotting/
else
    echo "Execution of original_powerlaw_mcmc.py failed."
fi

