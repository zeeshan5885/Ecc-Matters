import numpy as np
import os
from scipy.stats import truncnorm
from scipy.stats import norm
# Directory paths
input_directory = "./unweighted"  # Replace with the path to the input directory
output_directory = "./toy_events"  # Replace with the path to the output directory

# Get a list of all .dat files in the input directory
input_files = [file for file in os.listdir(input_directory) if file.endswith(".dat")]

for input_file in input_files:
    # Load the data from the input .dat file
    input_path = os.path.join(input_directory, input_file)
    data = np.loadtxt(input_path)

    # Extract the values from each column
    d1 = data[0]
    d2 = data[1]
    d3 = data[2]
    size = 4000
    #print("d1: ", d1, "d2: ", d2, "d3: ", d3)
    m1 = norm.rvs(loc=d1, scale=5, size=size)
    m2 = norm.rvs(loc=d2, scale=5, size=size)
    ecc = truncnorm.rvs(0, 1, loc=d3, scale=0.05, size=size)
    output_data = np.column_stack((m1, m2, ecc))
    col_names = ["m1_source", "m2_source", "ecc"]
    
    # Get the output file name by replacing the extension of the input file
    output_file = os.path.splitext(input_file)[0] + ".dat"
    output_path = os.path.join(output_directory, output_file)

    # Save the output data to the output file
    np.savetxt(output_path, output_data, delimiter='\t', header="\t".join(col_names))
