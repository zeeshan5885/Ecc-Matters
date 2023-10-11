import os

import matplotlib.pyplot as plt
import numpy as np
import RIFT.lalsimutils as lalsimutils
from scipy.stats import truncnorm

# Directory paths
input_directory = "./weighted_events"  # Replace with the path to the input directory
output_directory = "./rPE_events"  # Replace with the path to the output directory

# Get a list of all .dat files in the input directory
input_files = [file for file in os.listdir(input_directory) if file.endswith(".dat")]

for input_file in input_files:
    # Load the data from the input .dat file
    input_path = os.path.join(input_directory, input_file)
    data = np.loadtxt(input_path)

    # Extract the values from each column
    m1 = data[:, 0]
    m2 = data[:, 1]
    d3 = data[0, 2]

    print("d1: ", m1, "d2: ", m2, "d3: ", d3)

    ecc = truncnorm.rvs(0, 1, loc=d3, scale=0.05, size=len(m1))
    output_data = np.column_stack((m1, m2, ecc))
    col_names = ["m1_source", "m2_source", "ecc"]

    # Get the output file name by replacing the extension of the input file
    output_file = os.path.splitext(input_file)[0] + ".dat"
    output_path = os.path.join(output_directory, output_file)

    # Save the output data to the output file
    np.savetxt(output_path, output_data, delimiter="\t", header="\t".join(col_names))

# Finding the Median and standard deviation in each colum of all events

# Folder path containing the .dat files
