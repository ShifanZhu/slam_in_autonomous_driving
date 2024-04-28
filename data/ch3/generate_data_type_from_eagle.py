import heapq

def read_pose_file(filename):
    # Reads the ground truth pose file
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # Convert microseconds to seconds
            time_in_seconds = float(parts[0])
            # Keep other parts unchanged
            data = ' '.join(parts[1:])
            # Add the label "MoCap" and yield with the timestamp in seconds
            yield (time_in_seconds, f'MoCap {time_in_seconds} {data}')

def read_imu_file(filename):
    # Reads the IMU data file
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # Convert microseconds to seconds
            time_in_seconds = float(parts[0])
            # Keep other parts unchanged
            data = ' '.join(parts[1:])
            # Add the label "IMU" and yield with the timestamp in seconds
            yield (time_in_seconds, f'IMU {time_in_seconds} {data}')

def merge_data(pose_file, imu_file):
    # Creates a min-heap to merge data from both files by time in seconds
    merged_data = []
    # Use a heap to automatically sort entries by the first item in each tuple (time in seconds)
    heapq.heapify(merged_data)
    for data in read_pose_file(pose_file):
        heapq.heappush(merged_data, data)
    for data in read_imu_file(imu_file):
        heapq.heappush(merged_data, data)

    # Extract sorted data from the heap and return
    while merged_data:
        yield heapq.heappop(merged_data)

def main():
    pose_file = 'mh01_gt_data_sad.txt'
    imu_file = 'mh01_imu_data_sad.txt'
    output_file = 'combined_data.txt'

    # Use merge_data function to get sorted data and write to an output file
    with open(output_file, 'w') as file:
        for _, data in merge_data(pose_file, imu_file):
            file.write(f'{data}\n')
    print(f"Data combined and written to {output_file}")

if __name__ == "__main__":
    main()
