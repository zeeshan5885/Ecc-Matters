#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 21:33:41 2023

@author: mzeeshan
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
# Get a list of all files matching the pattern
file_list = glob.glob("./ecc_events/event_*.dat")

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
plt.savefig("scatter_plot.png")
plt.figure()


# Plotting the whole population with median and std of each event
#combined_data = np.loadtxt("mean_std.dat")
#lines={'linestyle': 'None'}
#plt.rc('lines', **lines)
#plt.xlabel("$m_1 [M_\odot]$")
#plt.ylabel("$m_2 [M_\odot]$")
#plt.plot(combined_data[:,0], combined_data[:,1], 'ro', markersize=10)
#plt.savefig("mean_masses.png")
#plt.figure()

# making the complete population plots
# Read the data from the text file
data = np.loadtxt("population.dat")
# Extract the columns
m1 = data[:, 0]
m2 = data[:, 1]
ecc = data[:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data points as dots with colors
sc = ax.scatter(m1, m2, ecc, c=ecc, cmap='plasma', marker='o')

# Set labels for the axes
ax.set_xlabel('$m_1[M_\odot]$')
ax.set_ylabel('$m_2[M_\odot]$')
ax.set_zlabel('$\epsilon$')

# Set a title for the plot
#ax.set_title('Eccentric Synthetic Population')

# Add interactivity with mplcursors
cursors = mplcursors.cursor(sc, hover=True)
cursors.connect("add", lambda sel: sel.annotation.set_text(f"({sel.target[0]:.2f}, {sel.target[1]:.2f}, {sel.target[2]:.2f})"))

# Add a colorbar
cbar = fig.colorbar(sc)
cbar.set_label('$\epsilon$')
plt.savefig('pop3dcom_0.05.png', bbox_inches='tight')
# Display the plot
plt.figure()


# making the weighted population plots
# Read the data from the text file
data = np.loadtxt("weighted_population.dat")
# Extract the columns
m1 = data[:, 0]
m2 = data[:, 1]
ecc = data[:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data points as dots with colors
sc = ax.scatter(m1, m2, ecc, c=ecc, cmap='plasma', marker='o')

# Set labels for the axes
ax.set_xlabel('$m_1[M_\odot]$')
ax.set_ylabel('$m_2[M_\odot]$')
ax.set_zlabel('$\epsilon$')

# Set a title for the plot
#ax.set_title('Eccentric Synthetic Population')

# Add interactivity with mplcursors
cursors = mplcursors.cursor(sc, hover=True)
cursors.connect("add", lambda sel: sel.annotation.set_text(f"({sel.target[0]:.2f}, {sel.target[1]:.2f}, {sel.target[2]:.2f})"))

# Add a colorbar
cbar = fig.colorbar(sc)
cbar.set_label('$\epsilon$')
plt.savefig('pop3d_0.05.png', bbox_inches='tight')
# Display the plot
plt.figure()

scaled_data = np.loadtxt("scaled_population.dat")
scaled_m1 = scaled_data[:,0]
scaled_m2 = scaled_data[:,1]
plt.plot(m1,m2,'o',label = 'EBBHs')
plt.plot(scaled_m1,scaled_m2,'.',label = 'NEBBHs')
plt.legend()
plt.xlabel('$m_1 [M_\odot]$')
plt.ylabel('$m_2[M_\odot]$')
#plt.title('scaled m1 vs scaled m2')
plt.savefig('pop2d_0.05.png', bbox_inches='tight')
plt.figure()
