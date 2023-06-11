'''
We are creating this file to generate the synthetic population given the fixed paramters of the powerlaw model defined in the paper.
The power law model which is defined as 'prob.joint' function in the syn_pop_prob.py file, we are importing this fucntion  here and generating the random events by putting the fix parameters of the population model. suh as N=number of events, alpha = power law factor, m_min, m_max, and M_max = m1+m2 : we are impsing uper bound 200 for the whole mass of the binary due to astrophysical reasons. 

To get any desired non spinining population, you can just change the paramter values in the follwing fucntion. 
'''
import numpy as np
import syn_pop_prob

# syn_pop_prob.joint_rvs(N,alpha,m_min,m_max,M_max=m1+m2,sigma_e)

array = syn_pop_prob.joint_rvs(100000,-3,10,50,100,0.2)


col_names = ["m1_source", "m2_source", "ecc"]

#saving array in a single file

np.savetxt("population.dat", array, header="\t".join(col_names))

# saving each row into a seprate file

#for i in range(array.shape[0]):
#    filename = f"row_{i+1}.txt"
#    row = array[i,:]
#    np.savetxt(filename, [row], delimiter="\t", fmt="%.3f",header="\t".join(col_names))

# Saving each row of the array to a separate file with column names and copying the row 4000 times

#for i in range(array.shape[0]):
#    filename = f"event_{i+1}.txt"
#    row = array[i,:]
#    data = np.repeat([row], 4000, axis=0)
#    np.savetxt(filename, data, delimiter="\t", header="\t".join(col_names))

#print(array)
