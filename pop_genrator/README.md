* This directory is solely created to generate the synthetic population by using th power law model including eccentricity. 
To generate the population, you just need to run the bash script named 'pop genrator.sh'. 

* This script will  auto delete the previously existed data and will create two fresh directories called *dat_evetns*, *txt_events* and one file with complete population nanamed combine _pop.txt. These two directoreis contain the same population but with different formate.

* As this model create the population with eccentricity, you can scale those events by using the equation (1.1) in the following paper. After scaling, you will get the population events without eccentricty and scaled mass.

* https://arxiv.org/abs/2108.05861 (Transforming equation)

* To get the scaled events, just run the run scaling.sh script and you will get the new directory with name scaled events which will contain all the scaled events.

* In addition to the scaled events, this run will also tell you about any missing events after removing the eccentricity. 
