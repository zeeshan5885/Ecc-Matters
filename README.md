* This code is Based on https://git.ligo.org/daniel.wysocki/bayesian-parametric-population-models/-/blob/master/src/pop_models/powerlaw/mcmc.py
* We modified this code to add the eccentricity parameter in the power law model.
* Example: You may want to look at the example of this code but without the eccentricity parameter. https://gitlab.com/dwysocki/pop-models-examples

# How to run this code to get the population inference

* First, create a virtual environment to install the required packages using the following command. or following instructions in the given link
https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment
```
python3 -m venv env
```
* Before starting the installation of the required packages, activate the package.
```
source env/bin/activate
```
* Now, you can use the pip install to load the required packages in the environment.
* The following packages need to install.
```
pip install RIFT
pip install mplcursors
pip install mpmath
pip install emcee
```
* You can play with the code after installing the required packages in the virtual environment.

* First, create the appropriate vt_m1_m2.hdf5 and copy it into the **population-generator**. (you will need it for weighting the synthetic population). You may use the already generated vt_m1_m2.hdf5 file, or you can use the following command to generate a new one according to your requirements.
* Syntax: Python vt.py m_min m_max N_samples duration_in_days output_file_name.hdf5
```
python vt.py 10 50 100 4 vt_m1_m2.hdf5
```
* Second, you need data containing three columns: m1_source, m2_source, ecc.
* TO check the code runs fine, you can use the sample.dat file in the events directory and run the following command.
```
./run.sh
```
If the above commands run without error and produce readable myout.hdf5 file, you are good to go.

* you can generate a desired population using the directory **population-genrator**. Please follow the steps written in README.md in that directory.
* if you create a synthetic population by using **population-generator** then generated events in directories **ecc_evetns** and **scaled_evenst** will be auto-moved to the appropriate directory to run. 
* Once you have the events in a directory and vt_m1_m2.hdf5 file, you can run the script. nohup will put the run in the background, and you can keep working in the terminal.
*  **Warning: ** Remember to delete the sample.dat file in the directories (weighted_events, scaled_events) before running the code for synthetic population or real PE.
```
nohup ./run.sh &
```
* By default in this script, we have set the number of walkers for MCMC method equal to 20 and the number of steps for each walker equal to 400. But you may change these as you like in the script.
* The run time will take hours to days, depending on data, walker, steps, and your machine.
* Meanwhile, start the second run for the data without eccentricity in **original_pop_code** directory, similar to with eccentricity. 
* Once both runs are done, you will get the myout.hdf5 and myout_org.hdf5 files containing the posterior for each parameter and you can make corner and chain plots by running the Python files in **plotting** directory.
```
python3 chain-plots.py
python3 corner-plots.py
```
* Remember that our power law model has three random variables (m1,m2,ecc) that we provide to the model through real or synthetic data.
* Then this code constrains the parameters of the population model: log rate, alpha, m_min, m_max, sigma_ecc.
