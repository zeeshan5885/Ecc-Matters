
* This code is Based on https://git.ligo.org/daniel.wysocki/bayesian-parametric-population-models/-/blob/master/src/pop_models/powerlaw/mcmc.py
* We modified this code to add the eccentircity parameter in the power law model.
* Example: You may want to look at the example of this code but without eccentricity paramter. https://gitlab.com/dwysocki/pop-models-examples

# How to run this code to get the population inference

* First, build the appropriate vt_m1_m2.hdf5 and copy it into population-genrator. (you will need it for weighting the synthetic populaiton)
```
python vt.py 10 50 100 4 vt_m1_m2.hdf5
```
* Second, you need a data containing three colums: m1_source , m2_source, ecc.
* You may make a fake data to check that if code actually works or not. For sample, fake.dat is given in repo. If it runs without error, you are good to go.
* You can also generate a desired population by using the directory **population-genrator**. Please follow the steps written in README.md in that directory.
* if you creates a sythetic population by using **population-genrator** then you need to move those **weighted_events** into **syn_events** directory.
* Once you have the evetns in syn_events direcory and vt_m1_m2.hdf5 file, you can run the script **run_syn_new.sh**.
```
nohup ./run_syn_new.sh &
```
* By default in this script, we have set the number of input files = 100, number of walker for mcmc method equal to 10000, and number of steps for each walker equal to 100. But you may change these as you like in the script.
* The run time will take hours to days depends on data, walker, steps, and your machine.
* Once the run is done, you will get the myout.hdf5 file which will contain the PE for each parameters and you can make corner plots as you need for your analysis.

* Keep in mind that our power law model has the three random variables (m1,m2,ecc) which we provide to the model thorugh the real data or synthetic data.
* Then this model gives us the inference for the five paramters of the model : log rate, alpha, m_min, m_max, sigma_ecc.
* Lastly, if you want to run the code without eccentricity and want to comapre the results with and without ecccentricity, then you may go into the **original-pop-code** direcotry and run this code in the similar method defined above. But keep in mind that this code requires data without ecccentricity and same vt_m1_m2.hdf5 file.
