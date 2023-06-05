* This directory is solely created to generate the synthetic population by using the eccentric power law model. 
To generate the population, you need to run the script as **./run_complete.sh** 

* This script will delete the previously existed file **population.dat** and will create new **population.dat** file which will contain 10000 events be default. In addition to single population file, it will create the two directories called **weighted_events** and **scaled_events**. The weighted directory will contain the 100 weighted events with VT files out of 10000. Afterwards, we will will use equation 1.1 in the following paper to scale those events and remove the ecentricity.
* 
* https://arxiv.org/abs/2108.05861 (Scaling equation 1.1)

* In addition to the scaled events, this run will also count and identify any missing events after scaling. Because, high eccentric events tends to miss after scaling.

* By default, this code will generate the population with set parameters **N= 10000 , Alpha = -2 , m-min = 10 , m-max = 50 , M= 100 , sigma_ecc =0.2.**

* So, to get a desired population, you need to edit the parameters in the **genrating.py** file. Particualraly, the arguments of the defined function **syn_pop_prob.joint_rvs(10000,-2,10,50,100,0.2)**

* The **generating.py** only creates the 10000 random events from the power_law_model.
* The **weighting.py** will choose the 100 weighted events out of 1000. These 100 are those which are well suited to be observed with LVK or you can say they may be in the sensitivity range of the LVK.
* Finally, **scaling.py** will remove the eccentricity form those 100 events will give us scaled events without eccentricity.
