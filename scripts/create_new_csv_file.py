import os
import csv
from datetime import datetime

def create_new_csv_file(base_folder="test_data", base_filename="test"):
    # Create the folder if it doesn't exist
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    # Get a list of existing files that match the pattern
    existing_files = [f for f in os.listdir(base_folder) if f.startswith(base_filename) and f.endswith('.csv')]
    
    # Extract the numbers from existing filenames
    numbers = []
    for file in existing_files:
        try:
            # Extract the number between base_filename and .csv
            number_part = file[len(base_filename):].split('.')[0]
            if number_part:
                numbers.append(int(number_part))
        except ValueError:
            continue
    
    # Determine the next number
    next_number = 1 if not numbers else max(numbers) + 1
    
    # Create the new filename
    new_filename = f"{base_filename}{next_number}.csv"
    new_filepath = os.path.join(base_folder, new_filename)
    
    # Create the new CSV file with a header row and timestamp
    with open(new_filepath, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Timestamp', 'Data Column 1', 'Data Column 2'])  # Example header
        csv_writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Initial Entry', 'Program Started'])
    
    print(f"Created new CSV file: {new_filepath}")
    return new_filepath

# Usage
if __name__ == "__main__":
    # Replace self.csv_file with the path to the new CSV file
    csv_file_path = create_new_csv_file()
    
    # You can now use csv_file_path in your program
    print(f"Using CSV file: {csv_file_path}")