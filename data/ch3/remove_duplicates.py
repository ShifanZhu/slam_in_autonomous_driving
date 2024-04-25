import os

def remove_timestamp_duplicates(file_path):
    # Initialize a list to maintain the order of lines and a set to track seen timestamps
    ordered_lines = []
    seen_timestamps = set()

    # Read the contents of the original file
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(maxsplit=1)  # Split only on the first space
            if len(parts) == 2:
                timestamp = parts[0]
                # Check if the timestamp has already been seen
                if timestamp not in seen_timestamps:
                    seen_timestamps.add(timestamp)
                    ordered_lines.append(line)

    # Write the non-duplicate, ordered lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(ordered_lines)

def process_directory(directory):
    # Walk through all subdirectories and files in the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                remove_timestamp_duplicates(file_path)

# Specify the root directory to start from
root_directory = "."  # Adjust this path to your needs

# Process all .txt files in the directory to remove duplicate lines based on timestamps while maintaining the original order
process_directory(root_directory)
