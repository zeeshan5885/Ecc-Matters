import numpy as np
import matplotlib.pyplot as plt
import h5py
import corner

data1 = h5py.File('myout.hdf5','r')
#data2 = h5py.File('myout.hdf5','r')
data_3D1 = np.array(data1['pos'])
#data_3D2 = np.array(data2['pos'])
print(data_3D1.shape)
data1_burn = data_3D1[1000:]
#data2_burn = data_3D2[1000:]
print(data1_burn.shape)

data_new1 = data1_burn.reshape((100000,5))
#data_new2 = data2_burn.reshape((50000,4))
#data_3D1_col_rm = data_new1[:,:4]
#print(data_new1)
#print(data_new2)
#print(np.sort(data_new1[:,-1]))
#plt.plot(np.sort(data_new1[:,-1]))

labels=[r"$log_{10}(\frac{\mathcal{R}}{Gpc^{-3}yr^-1})$", r"$\alpha$", r"$m_{min} [M_\odot$]",r"$m_{max} [M_\odot]$",
        "$\sigma_\epsilon$"]

#limits = [(1.2, 2.8),(-6,2),(0,15),(40,90),(0,0.1)]
# Calculate the mean values along each dimension
#mean_values1 = np.mean(data_new1, axis=0)
#provide true values if any
truth_values = [1, -2, 10, 50, 0.05]
#plotting the corner plot
figure1 = corner.corner(data_new1,labels=labels
                      ,show_titles=True,plot_datapoints=False,color='orange',
                        truths=truth_values,truth_color='red',smooth=True)
figure1.savefig("cor_ecc.png")
plt.show()
#labels0=[r"$log_{10}(\frac{\mathcal{R}}{Gpc^{-3}yr^-1})$", r"$\alpha$", r"$m_{min} [M_\odot]$",r"$m_{max} [M_\odot]$"]
#limits0 = [(1.2, 2.8),(-6,2),(0,12),(40,90)]

#truth_values1 = [2, -2, 10, 50]
#plotting the corner plot
#figure2 = corner.corner(data_new2,labels=labels0
#                      ,show_titles=False,plot_datapoints=False,color='orange',
#                        truths=truth_values1,truth_color='red')
#figure2.savefig("cor_org.png")    


# Create an empty column (filled with None values)
#new_column = np.full((2000, 1000, 1), None, dtype=object)
#new_column = np.zeros((2000, 1000, 1))
#new_column = np.random.randint(0, 2, size=(2000, 1000, 1))

# Concatenate the new column with the 3D array along the last axis (axis=2)
#data_3D2_col_add = np.concatenate((data_3D2, new_column), axis=2)
#data_new3 = data3_burn.reshape((1500000,5))
#print(data_3D2_col_add)

#figure4 = corner.corner(data_new3,labels=labels
#                      ,show_titles=True,plot_datapoints=False,color='orange',
#                        truths=truth_values,truth_color='red',smooth=True)
#figure4.savefig("cor_com.png")

#corner.corner(data_new3,fig=figure1,labels=labels
#                      ,show_titles=False,plot_datapoints=False,color='black'
#              )
#figure1.savefig("com_0.2_ecc.png")