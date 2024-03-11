import numpy as np
import os
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import RIFT.lalsimutils as lalsimutils

# Directory paths
input_directory = "./weighted_events"  # Replace with the path to the input directory
output_directory = "./ecc_events"  # Replace with the path to the output directory

# Get a list of all .dat files in the input directory
input_files = [file for file in os.listdir(input_directory) if file.endswith(".dat")]

for input_file in input_files:
    # Load the data from the input .dat file
    input_path = os.path.join(input_directory, input_file)
    data = np.loadtxt(input_path)

    # Extract the values from each column
    d1 = data[0, 0]
    d2 = data[0, 1]
    d3 = data[0, 2]

    #print("d1: ", d1, "d2: ", d2, "d3: ", d3)

    # Compute the M_chirp from component masses
    Mc_true = (d1 * d2) ** (3./5.) / (d1 + d2) ** (1./5.)
    # Compute symmetric mass ratio (Eta) from component masses
    eta_true = (d1 * d2) / (d1 + d2)**2

    #print("Mc_True: ", Mc_true, "eta_true: ", eta_true)

    # Adding Errors in True M chirp and Eta according to paper
    row = 9 /np.power(np.random.uniform(), 1./3.) #SNR
    
    v_PN_param = (np.pi* Mc_true*20*lalsimutils.MsunInSec)**(1./3.)  # 'v' parameter
    v_PN_param_max = 0.2
    v_PN_param = np.min([v_PN_param,v_PN_param_max])
    snr_fac = row/15
    ln_mc_error_pseudo_fisher = 1.5*0.3*(v_PN_param/v_PN_param_max)**(7.)/snr_fac  # this ignores range due to redshift / distance, based on a low-order esti

#    print(ln_mc_error_pseudo_fisher, v_PN_param,Mc_true)
    alpha = np.min([0.07/snr_fac, ln_mc_error_pseudo_fisher])  # Percentage error, we are adding 10%.  Note already accounts for SNR effects
#    print(alpha)
    # Shifting the mean using standard normal distribution
    ro = np.random.normal(0, 1)
    rop = np.random.normal(0, 1)
    #print("ro: ", ro, "rop: ", rop)

    size = 20000
    std_dev = 1
    #print("std_dev: ", std_dev)
    
    # Changing the width across the shifted mean
    r = np.random.normal(0, std_dev, size)
    rp = np.random.normal(0,std_dev, size)
    
    # Defining the relation
    Mc = Mc_true * (1 + alpha * (ro + r))
    eta = eta_true * (1 + 0.03 * (8 / row) * (rop + rp))

    #print("Mc_array: ", Mc, "eta_array: ", eta)

    # Compute component masses from Mc, eta. Returns m1 >= m2
    etaV = np.array(1 - 4 * eta, dtype=float)
    if isinstance(eta, float):
        if etaV < 0:
            etaV = 0
            etaV_sqrt = 0
        else:
            etaV_sqrt = np.sqrt(etaV)
    else:
        indx_ok = etaV >= 0
        etaV_sqrt = np.zeros(len(etaV), dtype=float)
        etaV_sqrt[indx_ok] = np.sqrt(etaV[indx_ok])
        etaV_sqrt[np.logical_not(indx_ok)] = 0  # Set negative cases to 0, so no sqrt problems
    m1 = 0.5 * Mc * eta ** (-3. / 5.) * (1. + etaV_sqrt)
    m2 = 0.5 * Mc * eta ** (-3. / 5.) * (1. - etaV_sqrt)
    ecc = truncnorm.rvs(0, 1, loc=d3, scale=0.1, size=size)
    output_data = np.column_stack((m1, m2, ecc))
    print(input_file, np.sum(indx_ok))
    output_data = output_data[indx_ok,:]
    col_names = ["m1_source", "m2_source", "ecc"]
    
    # Get the output file name by replacing the extension of the input file
    output_file = os.path.splitext(input_file)[0] + ".dat"
    output_path = os.path.join(output_directory, output_file)

    # Save the output data to the output file
    np.savetxt(output_path, output_data, delimiter='\t', header="\t".join(col_names))

# Finding the Median and standard deviation in each colum of all events

# Folder path containing the .dat files
folder_path = "./ecc_events"

# Initialize lists to store column statistics
all_medians = []
all_std_deviations = []

# Iterate through each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".dat"):
        file_path = os.path.join(folder_path, file_name)
        
        # Load data from the file
        data = np.loadtxt(file_path)
        
        # Calculate median and standard deviation for each column
        file_medians = np.median(data, axis=0)
        file_std_deviations = np.std(data, axis=0)
        
        # Append column statistics to the overall lists
        all_medians.append(file_medians)
        all_std_deviations.append(file_std_deviations)

# Combine medians and std_deviations into a single array
combined_data = np.hstack((np.vstack(all_medians), np.vstack(all_std_deviations)))
col_names_ms = ["m1_mean", "m2_mean", "ecc_mean","m1_std","m2_std","ecc_std"]
# Save combined_data to a new .dat file
output_file_path = "./mean_std.dat"
np.savetxt(output_file_path, combined_data, delimiter="\t", fmt="%.6f",header="\t".join(col_names_ms))

