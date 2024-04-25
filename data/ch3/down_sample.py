def save_every_nth_line(source_file, output_file, n):
    """
    Reads a file and writes every nth line to another file.
    
    Parameters:
        source_file (str): Path to the source file.
        output_file (str): Path to the output file.
        n (int): The step size (n) to save lines; saves 1 line every n lines.
    """
    try:
        with open(source_file, 'r') as file:
            with open(output_file, 'w') as outfile:
                # Enumerate over each line in the file, starting at 1
                for i, line in enumerate(file, 1):
                    # Check if the current line number is one that should be saved
                    if i % n == 0:
                        outfile.write(line)
        print(f"Successfully saved every {n}th line from '{source_file}' to '{output_file}'.")
    except FileNotFoundError:
        print(f"Error: The file '{source_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ask the user to specify the interval n
    n = int(input("Enter the value of n to save every nth line: "))
    # Define the source and destination files
    source_file = 'E-ATS-IMU.txt'
    output_file = 'E-ATS-IMU_filter.txt'
    
    # Call the function with user-defined n
    save_every_nth_line(source_file, output_file, n)

