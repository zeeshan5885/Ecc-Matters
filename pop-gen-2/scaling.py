import numpy as np
import os

# Directory containing the input files
input_dir = './txt_events/'

# Directory to save the output files
output_dir = './scaled_events/'
col_names = ["m1_source", "m2_source", "ecc"]
# Iterate over the text files
for i in range(1, 101):
    # Generate the file path for the current iteration
    file_path = os.path.join(input_dir, f'event_{i}.txt')

    # Load the data from the text file
    data = np.loadtxt(file_path)

    # Extract the columns
    col1 = data[:, 0]
    col2 = data[:, 1]
    col3 = data[:, 2]
    scale_factor = ((1-(157/24)*col3**2)**(3/5))
    # Scaling equation: scaled_value = column_value * scale_factor
    scaled_col1 = col1 * scale_factor
    scaled_col2 = col2 * scale_factor

    # Combine the scaled columns back into a single array
    scaled_data = np.column_stack((scaled_col1, scaled_col2, col3))

    # Generate the output file path for the current iteration
    output_file_path = os.path.join(output_dir, f'event_{i}.txt')

    # Save the scaled data to a new text file
    #np.savetxt(output_file_path, scaled_data, header='m1_source m2_source ecc', comments='')
    np.savetxt(output_file_path, scaled_data, delimiter="\t", fmt="%.3f", header="\t".join(col_names))



#finding the values which have nan values for masses after conversion  and printing their eccentricity value


import numpy as np
import os

# Directory containing the text files
folder_path = './scaled_events/'

# Counter for the files with NaN or zero values
count = 0

# Iterate over the files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(folder_path, file_name)

        # Load the data from the text file
        data = np.loadtxt(file_path)

        # Check if any NaN or zero values exist in the data
        if np.isnan(data).any() or np.count_nonzero(data == 0) > 0:
            count += 1
            print(f"File '{file_name}' contains NaN or zero values.")

            # Print the minimum value in the file
            min_value = np.nanmin(data)
            print(f"Minimum value in '{file_name}': {min_value}")

# Print the count of files with NaN or zero values
print(f"Total files with NaN or zero values: {count}")


# Load the data from the text file and scaling the single txt file which have the complete population
data01 = np.loadtxt('combine_pop.txt')

# Extract the columns
col01 = data01[:, 0]
col02 = data01[:, 1]
col03 = data01[:, 2]
scale_factor01 = ((1-(157/24)*col03**2)**(3/5))


# Scaling equation: scaled_value = column_value * scale_factor
scaled_col01 = col01 * scale_factor01
scaled_col02 = col02 * scale_factor01

# Combine the scaled columns back into a single array
scaled_data01 = np.column_stack((scaled_col01, scaled_col02, col03))

# Save the scaled data to a new text file
np.savetxt('combine_scaled_pop.txt', scaled_data01)

