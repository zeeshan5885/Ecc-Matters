* This directory is solely created to generate the synthetic population by using the eccentric power law model. 
To generate the population, you need to run the script as **./run_complete.sh** 

* By default, this code will generate the population with set parameters **$N= 10000$ , $\alpha = -2$ , $m_{min} = 10$ , $m_{max} = 50$ , $M = 100$ , $\sigma_\epsilon =0.2$.**

* So, to get a desired population, you need to edit the parameters in the **genrating.py** file. Particualraly, the arguments of the defined function **syn_pop_prob.joint_rvs(10000,-2,10,50,100,0.2)**

* First,  **generating.py**  creates the 10000 random events from the power_law_model with eccentric
ity.

* Second, The **weighting.py** choose the 100 weighted events out of 10000. These 100 are those which are well suited to be observed with LVK or you can say they may be in the sensitivity range of the LVK.
* Third, **real_PE.py** takes the masses and eccentriciy to add gaussian uncertainites in each of them.
* Finally, **scaling.py** removes the eccentricity form those 100 events and give us scaled events without eccentricity using the equation 1.1 given in the paper. 
* https://arxiv.org/abs/2108.05861 (Scaling equation 1.1)

* In addition to the scaled events, this run will also count and identify any missing events after scaling. Because, high eccentric events tends to miss after scaling.
