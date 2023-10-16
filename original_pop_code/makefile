all:
	time python3 -mcProfile original_powerlaw_mcmc.py scaled_events/*.dat vt_m1_m2.hdf5 myout_org.hdf5 --n-walkers 20 --n-samples 500
	mv myout_org.hdf5 ../plotting/

rename:
	cp vt_m1_m2.hdf5 vt_m1_m2.hdf5.bak