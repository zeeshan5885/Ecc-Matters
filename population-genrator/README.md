* This directory is created solely to generate the synthetic population using the eccentric power law model. 
To generate the population, you just need to run the script as follows and it will generate a population with and without eccentricity after scaling. 
```
./run_complete.sh 
```
* By default, this code will generate the population with set parameters **$N= 10000$ , $\alpha = -2$ , $m_{min} = 10$ , $m_{max} = 50$ , $M = 100$ , $\sigma_\epsilon =0.05$.**

* To get a desired population, edit the parameters in the **genrating.py** file. Particularly the arguments of the defined function **syn_pop_prob.joint_rvs(10000,-2,10,50,100,0.2)**

* First,  **generating.py**  creates the 10000 random events from the power_law_model with eccentricity.

* Second, The **weighting.py** chooses the 100 weighted events out of 10000. These 100 are those which are well suited to be observed with LVK, or you can say they may be in the sensitivity range of the LVK.
* Third, **real_PE.py** takes the masses and eccentricity to add Gaussian uncertainties in each of them.
* Finally, **scaling.py** removes the eccentricity from those 100 events and gives us scaled events without eccentricity using equation 1.1 given in the paper. 
* https://arxiv.org/abs/2108.05861 (Scaling equation 1.1)

* Besides the scaled events, this run will count and identify any missing events after scaling. Because high eccentric events tend to miss after scaling.
