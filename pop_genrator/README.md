* This directory is solely created to generate the synthetic population by using the eccentric power law model. 
To generate the population, you just need to run the bash script named **pop-genrator.sh** 

* This script will  auto delete the previously existed data and will create two fresh directories called **dat-evetns**, **txt-events** and one file with whole population named **combine-pop.txt**. These two directoreis contain the same population but with different format.

* As this model creates the population with eccentricity, you can scale (removing the role of eccentricity from the events and getting the lesser mass of BHs) those events by using the equation (1.1) in the following paper.

* https://arxiv.org/abs/2108.05861 (Scaling equation 1.1)

* To get the scaled events, just run the **run-scaling.sh** script and you will get the new directory with the name scaled-events.

* In addition to the scaled events, this run will also count and identify any missing events after scaling.

* By default, this code will generate the population with set parameters **N= 100 , Alpha = 2 , m-min = 10 , m-max = 50 , M= 200.**

* So to get the any desired population, you need to edit the parameters in the **population-gen.py** file. 
