'''
We are creating this file to generate the synthetic population given the fixed paramters of the powerlaw model defined in the paper.
'''
import numpy as np
import syn_pop_prob
array = syn_pop_prob.joint_rvs(100,-2,5,50,200)


col_names = ["m1_source", "m2_source", "ecc"]

#saving array in a single file

np.savetxt("combine_pop.txt", array, header="\t".join(col_names))

# saving each row into a seprate file

for i in range(array.shape[0]):
    filename = f"row_{i+1}.txt"
    row = array[i,:]
    np.savetxt(filename, [row], delimiter="\t", fmt="%.3f",header="\t".join(col_names))

# Saving each row of the array to a separate file with column names

for i in range(array.shape[0]):
    filename = f"event_{i+1}.txt"
    row = array[i,:]
    data = np.repeat([row], 4000, axis=0)
    np.savetxt(filename, data, delimiter="\t", fmt="%.3f", header="\t".join(col_names))

#print(array)
