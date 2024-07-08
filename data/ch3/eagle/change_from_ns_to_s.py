# Path to your input file
input_file_path = 'gt_pose.txt'
# Path to your output file
output_file_path = 'GT.txt'

# Open the input file and output file
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    # Iterate over each line in the file
    for line in input_file:
        # Split the line into timestamp and the rest of the line
        parts = line.split(maxsplit=1)
        if len(parts) >= 2:
            timestamp_ns, rest_of_line = parts
            try:
                # Convert the timestamp to seconds
                timestamp_s = float(timestamp_ns) * 1e-6
                # Write the new line with converted timestamp and the rest of the line
                output_file.write(f'{timestamp_s} {rest_of_line}')
            except ValueError:
                # Handle the case where the timestamp is not a valid number
                output_file.write(line)  # Optionally handle or log the error
        else:
            # If there's no second part, just write the line back
            output_file.write(line)

