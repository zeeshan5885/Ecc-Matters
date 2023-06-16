#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 21:33:41 2023

@author: mzeeshan
"""
import numpy as np
import matplotlib.pyplot as plt
import glob

# Get a list of all files matching the pattern
file_list = glob.glob("./rPE_events/event_*.dat")

# Iterate over each file
for i, file_path in enumerate(file_list):
    # Load data from the file
    data = np.loadtxt(file_path)
    m1 = data[:, 0]
    m2 = data[:, 1]
    ecc = data[:,2]

    # Scatter plot with different colors for each file
    plt.scatter(m1, m2, label=f"event {i+1}")

# Set plot title and labels
plt.title("Scatter Plot")
plt.xlabel("$m_1 [M_\odot]$")
plt.ylabel("$m_2 [M_\odot]$")
#plt.legend()  # Display legend
plt.show()
plt.savefig("scatter_plot.png")


# Plotting the whole population with median and std of each event
combined_data = np.loadtxt("mean_std.dat")
lines={'linestyle': 'None'}
plt.rc('lines', **lines)
plt.xlabel("$m_1 [M_\odot]$")
plt.ylabel("$m_2 [M_\odot]$")
plt.plot(combined_data[:,0], combined_data[:,1], 'ro', markersize=10)
plt.show()
plt.savefig("mean_masses.png")

