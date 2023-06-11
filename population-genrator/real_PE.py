import os
import numpy as np
from scipy.stats import truncnorm

# Specify the path to the folder containing the .dat files
folder_path = './weighted_events'

# Specify the path to the folder where the copies will be saved
output_folder = './rPE_events'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Set the number of copies to generate
num_copies = 4000

# Set the standard deviation factor for Gaussian uncertainty
std_dev_factor = 25
col_names = ["m1_source", "m2_source","ecc"]
# Iterate over all .dat files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        file_path = os.path.join(folder_path, filename)

        # Load the data from the .dat file
        data = np.loadtxt(file_path)

        # Extract the values from each column
        m1 = data[0, 0]
        m2 = data[0, 1]
        ecc = data[0, 2]
        # Initialize an empty array to store the copies
        copies = np.zeros((num_copies, 3))
        # Generate copies with Gaussian uncertainty for each column
        for i in range(num_copies):
            copies[i, 0] = np.random.normal(m1, std_dev_factor)
            copies[i, 1] = np.random.normal(m2, std_dev_factor)
            copies[i, 2] = truncnorm.rvs(0,1,loc=ecc,scale=0.15, size=1)
        copies[0,:] = m1,m2,ecc
        # Create the output file path and save the copies
        output_filename = os.path.splitext(filename)[0] + ".dat"
        output_path = os.path.join(output_folder, output_filename)
        np.savetxt(output_path, copies, delimiter='\t',header="\t".join(col_names))

