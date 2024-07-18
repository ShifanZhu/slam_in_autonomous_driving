def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            timestamp = float(parts[1])
            data.append((timestamp, line.strip()))
    return data

def write_combined_file(data, output_filename):
    with open(output_filename, 'w') as file:
        for _, line in sorted(data):
            file.write(line + '\n')

def combine_files(file1, file2, output_file):
    data1 = read_file(file1)
    data2 = read_file(file2)
    combined_data = data1 + data2
    write_combined_file(combined_data, output_file)

# File paths
file1 = 'imu_data_sad.txt'
file2 = 'feature_points.txt'
output_file = 'combined_data.txt'

# Combine the files
combine_files(file1, file2, output_file)

print("Files combined successfully into", output_file)

