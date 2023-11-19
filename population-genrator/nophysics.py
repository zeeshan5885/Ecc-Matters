import os

def read_and_store_dat(file_path, output_directory):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(file_path, 'r') as file:
        # Read the header
        header = file.readline().strip().split(',')

        # Read and store data rows
        for line_number, line in enumerate(file, start=1):
            # Create a filename based on line number
            output_file_path = os.path.join(output_directory, f'toy_{line_number}.dat')

            # Write the header to the first line of each data file
            with open(output_file_path, 'w') as output_file:
                output_file.write(','.join(header) + '\n')
                output_file.write(line)

    print(f'Data stored in separate files in {output_directory}')

# Example usage
dat_file_path = 'population.dat'
output_dir = './unweighted/'
read_and_store_dat(dat_file_path, output_dir)
