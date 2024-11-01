import numpy as np
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.mplot3d import Axes3D
import mplcursors

file_list = glob.glob("./ecc_events/event_*.dat")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for i, file_path in enumerate(file_list):
    data = np.loadtxt(file_path)
    m1 = data[:, 0]
    m2 = data[:, 1]
    ecc = data[:, 2]

    # Scatter plot with different colors for each file
    sc = ax.scatter(m1, m2, ecc, c=ecc, cmap='plasma',s=1, marker='.', label=f"Event File {i+1}", alpha=0.1)

# Load and plot the true points data
true_data_path = "./weighted_population.dat"  # Path to the true points file
true_data = np.loadtxt(true_data_path)
true_m1 = true_data[:, 0]
true_m2 = true_data[:, 1]
true_ecc = true_data[:, 2]

# Plot true points with a distinct color, marker, and higher zorder
ax.scatter(true_m1, true_m2, true_ecc, color='red', marker='*', s=20, label="True Points",alpha=1)

# Set labels and add color bar
ax.set_xlabel('m1')
ax.set_ylabel('m2')
ax.set_zlabel('ecc')
plt.colorbar(sc, ax=ax, label='Ecc')

# Save the figure
plt.savefig("scatter_plot_with_true_points.png")