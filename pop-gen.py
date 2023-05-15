'''
We are creating this file to generate the synthetic population given the fixed paramters of the powerlaw model defined in the paper.
'''

import syn_pop_prob
population = syn_pop_prob.joint_rvs(100,-2,5,50,200)

print(population)
