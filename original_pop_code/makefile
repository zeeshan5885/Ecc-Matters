run_mcmc:
	time python3 -mcProfile original_powerlaw_mcmc.py scaled_events/*.dat vt_m1_m2.hdf5 myout_org.hdf5 --n-walkers 20 --n-samples 500
	mv myout_org.hdf5 ../plotting/

run_vt:
	python vt.py 10 50 100 1 vt_m1_m2.hdf5

rename:
	cp vt_m1_m2.hdf5 vt_m1_m2.hdf5.bak