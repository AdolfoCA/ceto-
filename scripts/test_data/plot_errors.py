import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def read_csv_with_comments(filepath):
    """
    Read a CSV file that has comments starting with # and possibly multiple header sections.
    """
    # Read the file lines
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    # Find the actual data header line (looks for 'Time,Fx,Fy,Fz,')
    header_pattern = re.compile(r'Time,Fx,Fy,Fz,')
    data_start_idx = None
    
    for i, line in enumerate(lines):
        if header_pattern.search(line):
            data_start_idx = i
            break
    
    if data_start_idx is None:
        raise ValueError("Could not find the data section in the CSV file")
    
    # Extract PID values from comments if available
    pid_values = {}
    for line in lines[:data_start_idx]:
        if line.startswith('#'):
            # Try to extract PID values
            pid_match = re.search(r'#\s*(\w+):\s*kp=([0-9.]+),\s*ki=([0-9.]+),\s*kd=([0-9.]+)', line)
            if pid_match:
                control, kp, ki, kd = pid_match.groups()
                pid_values[control] = {
                    'kp': float(kp),
                    'ki': float(ki),
                    'kd': float(kd)
                }
    
    # Create a new list with just the header and data rows
    data_lines = [lines[data_start_idx]] + lines[data_start_idx + 1:]
    
    # Use pandas to read this filtered content
    import io
    df = pd.read_csv(io.StringIO(''.join(data_lines)))
    
    return df, pid_values

def plot_data(csv_file):
    """
    Read a CSV file and create various plots as specified.
    
    Parameters:
    csv_file (str): Path to the CSV file
    """
    # Read the CSV file
    try:
        df, pid_values = read_csv_with_comments(csv_file)
        print(f"Successfully read CSV file with {len(df)} rows")
        print(f"Columns found: {', '.join(df.columns)}")
        if pid_values:
            print("Found PID values in comments:")
            for control, values in pid_values.items():
                print(f"  {control}: kp={values['kp']}, ki={values['ki']}, kd={values['kd']}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check if required columns exist
    required_columns = ['Time', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 
                        'Pitch', 'Roll', 'Yaw', 'Altitude', 'x', 'y']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {', '.join(missing_columns)}")
    
    # Create output directory for plots
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert time to relative time in seconds (starting from 0)
    if 'Time' in df.columns and len(df) > 0:
        start_time = df['Time'].iloc[0]
        df['RelativeTime'] = df['Time'] - start_time
    else:
        df['RelativeTime'] = np.arange(len(df))
    
    # Calculate position errors if setpoints are available
    # For X position
    if 'x' in df.columns and 'set_x' in df.columns:
        df['error_x'] = df['set_x'] - df['x']
    # For Y position
    if 'y' in df.columns and 'set_y' in df.columns:
        df['error_y'] = df['set_y'] - df['y']
    # For Z position (or altitude)
    if 'z' in df.columns and 'set_z' in df.columns:
        df['error_z'] = df['set_z'] - df['z']
    elif 'Altitude' in df.columns and 'set_altitude' in df.columns:
        df['error_z'] = df['set_altitude'] - df['Altitude']
    
    # Calculate angular errors if setpoints are available
    # For Roll
    if 'Roll' in df.columns and 'set_roll' in df.columns:
        df['error_roll'] = df['set_roll'] - df['Roll']
    # For Pitch
    if 'Pitch' in df.columns and 'set_pitch' in df.columns:
        df['error_pitch'] = df['set_pitch'] - df['Pitch']
    # For Yaw
    if 'Yaw' in df.columns and 'set_yaw' in df.columns:
        df['error_yaw'] = df['set_yaw'] - df['Yaw']
    
    # 1. Plot F and position variables
    plt.figure(figsize=(12, 15))
    
    # Fx vs x and set_x
    if 'Fx' in df.columns and 'x' in df.columns:
        plt.subplot(3, 1, 1)
        plt.plot(df['RelativeTime'], df['Fx'], 'b-', label='Fx')
        plt.ylabel('Force Fx (N)', color='blue')
        plt.title('Fx vs x and setpoint')
        plt.grid(True)
        plt.legend(loc='upper left')
        
        plt2 = plt.twinx()
        plt2.plot(df['RelativeTime'], df['x'], 'r-', label='x')
        # Add setpoint if available
        if 'set_x' in df.columns:
            plt2.plot(df['RelativeTime'], df['set_x'], 'r--', label='set_x')
        plt2.set_ylabel('Position x (m)', color='red')
        plt2.tick_params(axis='y', labelcolor='red')
        plt2.legend(loc='upper right')
    
    # Fy vs y and set_y
    if 'Fy' in df.columns and 'y' in df.columns:
        plt.subplot(3, 1, 2)
        plt.plot(df['RelativeTime'], df['Fy'], 'b-', label='Fy')
        plt.ylabel('Force Fy (N)', color='blue')
        plt.title('Fy vs y and setpoint')
        plt.grid(True)
        plt.legend(loc='upper left')
        
        plt2 = plt.twinx()
        plt2.plot(df['RelativeTime'], df['y'], 'r-', label='y')
        # Add setpoint if available
        if 'set_y' in df.columns:
            plt2.plot(df['RelativeTime'], df['set_y'], 'r--', label='set_y')
        plt2.set_ylabel('Position y (m)', color='red')
        plt2.tick_params(axis='y', labelcolor='red')
        plt2.legend(loc='upper right')
    
    # Fz vs z/Altitude and set_altitude
    if 'Fz' in df.columns and 'z' in df.columns:
        plt.subplot(3, 1, 3)
        plt.plot(df['RelativeTime'], df['Fz'], 'b-', label='Fz')
        plt.ylabel('Force Fz (N)', color='blue')
        plt.xlabel('Time (s)')
        plt.title('Fz vs z and setpoint')
        plt.grid(True)
        plt.legend(loc='upper left')
        
        plt2 = plt.twinx()
        plt2.plot(df['RelativeTime'], df['z'], 'r-', label='z')
        # Add setpoint if available
        if 'set_z' in df.columns:
            plt2.plot(df['RelativeTime'], df['set_z'], 'r--', label='set_z')
        plt2.set_ylabel('Position z (m)', color='red')
        plt2.tick_params(axis='y', labelcolor='red')
        plt2.legend(loc='upper right')
    elif 'Fz' in df.columns and 'Altitude' in df.columns:
        # Use Altitude as a substitute for z if z is not available
        plt.subplot(3, 1, 3)
        plt.plot(df['RelativeTime'], df['Fz'], 'b-', label='Fz')
        plt.ylabel('Force Fz (N)', color='blue')
        plt.xlabel('Time (s)')
        plt.title('Fz vs Altitude and setpoint')
        plt.grid(True)
        plt.legend(loc='upper left')
        
        plt2 = plt.twinx()
        plt2.plot(df['RelativeTime'], df['Altitude'], 'r-', label='Altitude')
        # Add setpoint if available
        if 'set_altitude' in df.columns:
            plt2.plot(df['RelativeTime'], df['set_altitude'], 'r--', label='set_altitude')
        plt2.set_ylabel('Altitude (m)', color='red')
        plt2.tick_params(axis='y', labelcolor='red')
        plt2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'force_position_plots.png'))
    plt.close()
    
    # 2. Plot M and orientation variables
    plt.figure(figsize=(12, 15))
    
    # Mx vs Roll and set_roll
    if 'Mx' in df.columns and 'Roll' in df.columns:
        plt.subplot(3, 1, 1)
        plt.plot(df['RelativeTime'], df['Mx'], 'b-', label='Mx')
        plt.ylabel('Moment Mx (Nm)', color='blue')
        plt.title('Mx vs Roll and setpoint')
        plt.grid(True)
        plt.legend(loc='upper left')
        
        plt2 = plt.twinx()
        plt2.plot(df['RelativeTime'], df['Roll'], 'r-', label='Roll')
        # Add setpoint if available
        if 'set_roll' in df.columns:
            plt2.plot(df['RelativeTime'], df['set_roll'], 'r--', label='set_roll')
        plt2.set_ylabel('Roll (deg)', color='red')
        plt2.tick_params(axis='y', labelcolor='red')
        plt2.legend(loc='upper right')
    
    # My vs Pitch and set_pitch
    if 'My' in df.columns and 'Pitch' in df.columns:
        plt.subplot(3, 1, 2)
        plt.plot(df['RelativeTime'], df['My'], 'b-', label='My')
        plt.ylabel('Moment My (Nm)', color='blue')
        plt.title('My vs Pitch and setpoint')
        plt.grid(True)
        plt.legend(loc='upper left')
        
        plt2 = plt.twinx()
        plt2.plot(df['RelativeTime'], df['Pitch'], 'r-', label='Pitch')
        # Add setpoint if available
        if 'set_pitch' in df.columns:
            plt2.plot(df['RelativeTime'], df['set_pitch'], 'r--', label='set_pitch')
        plt2.set_ylabel('Pitch (deg)', color='red')
        plt2.tick_params(axis='y', labelcolor='red')
        plt2.legend(loc='upper right')
    
    # Mz vs Yaw and set_yaw
    if 'Mz' in df.columns and 'Yaw' in df.columns:
        plt.subplot(3, 1, 3)
        plt.plot(df['RelativeTime'], df['Mz'], 'b-', label='Mz')
        plt.ylabel('Moment Mz (Nm)', color='blue')
        plt.xlabel('Time (s)')
        plt.title('Mz vs Yaw and setpoint')
        plt.grid(True)
        plt.legend(loc='upper left')
        
        plt2 = plt.twinx()
        plt2.plot(df['RelativeTime'], df['Yaw'], 'r-', label='Yaw')
        # Add setpoint if available
        if 'set_yaw' in df.columns:
            plt2.plot(df['RelativeTime'], df['set_yaw'], 'r--', label='set_yaw')
        plt2.set_ylabel('Yaw (deg)', color='red')
        plt2.tick_params(axis='y', labelcolor='red')
        plt2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'moment_orientation_plots.png'))
    plt.close()
    
    # 3. Plot all thrusters
    thruster_cols = [col for col in df.columns if 'Thruster' in col]
    if thruster_cols:
        plt.figure(figsize=(12, 8))
        for col in thruster_cols:
            plt.plot(df['RelativeTime'], df[col], label=col)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Thruster Output (N)')
        plt.title('All Thrusters Output vs Time')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'thrusters_plot.png'))
        plt.close()
    else:
        print("No thruster columns found in the data")
    
    # 4. Plot PID values
    pid_cols = [col for col in df.columns if 'PID' in col]
    if pid_cols:
        plt.figure(figsize=(12, 8))
        for col in pid_cols:
            plt.plot(df['RelativeTime'], df[col], label=col)
        
        plt.xlabel('Time (s)')
        plt.ylabel('PID Value')
        plt.title('PID Values vs Time')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pid_values_plot.png'))
        plt.close()
    else:
        print("No PID columns found in the data")
        
        # If we have PID values from comments, display them
        if pid_values:
            plt.figure(figsize=(10, 6))
            controls = list(pid_values.keys())
            
            kp_values = [pid_values[c]['kp'] for c in controls]
            ki_values = [pid_values[c]['ki'] for c in controls]
            kd_values = [pid_values[c]['kd'] for c in controls]
            
            x = np.arange(len(controls))
            width = 0.25
            
            plt.bar(x - width, kp_values, width, label='kp')
            plt.bar(x, ki_values, width, label='ki')
            plt.bar(x + width, kd_values, width, label='kd')
            
            plt.xlabel('Control')
            plt.ylabel('Value')
            plt.title('PID Values from Comments')
            plt.xticks(x, controls)
            plt.legend()
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pid_values_from_comments.png'))
            plt.close()
    
    # 5. NEW: Plot position errors
    position_errors = ['error_x', 'error_y', 'error_z']
    if any(error in df.columns for error in position_errors):
        plt.figure(figsize=(12, 8))
        
        for error in position_errors:
            if error in df.columns:
                plt.plot(df['RelativeTime'], df[error], label=error)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Position Error (m)')
        plt.title('Position Errors vs Time')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'position_errors_plot.png'))
        plt.close()
    else:
        print("No position error data could be calculated (missing setpoints or position data)")
    
    # 6. NEW: Plot angular errors
    angular_errors = ['error_roll', 'error_pitch', 'error_yaw']
    if any(error in df.columns for error in angular_errors):
        plt.figure(figsize=(12, 8))
        
        for error in angular_errors:
            if error in df.columns:
                plt.plot(df['RelativeTime'], df[error], label=error)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Error (deg)')
        plt.title('Angular Errors vs Time')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'angular_errors_plot.png'))
        plt.close()
    else:
        print("No angular error data could be calculated (missing setpoints or angle data)")
        
    # 7. NEW: Plot combined error RMS over time
    # Calculate RMS of position errors if available
    position_error_cols = [col for col in ['error_x', 'error_y', 'error_z'] if col in df.columns]
    if position_error_cols:
        # Square each error
        for col in position_error_cols:
            df[f"{col}_squared"] = df[col]**2
        
        # Sum the squares
        df['position_error_sum_squared'] = df[[f"{col}_squared" for col in position_error_cols]].sum(axis=1)
        
        # Take the square root to get RMS
        df['position_error_rms'] = np.sqrt(df['position_error_sum_squared'])
        
        # Plot RMS position error
        plt.figure(figsize=(12, 6))
        plt.plot(df['RelativeTime'], df['position_error_rms'], 'b-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('RMS Position Error (m)')
        plt.title('Root Mean Square Position Error vs Time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'position_error_rms_plot.png'))
        plt.close()
    
    # Calculate RMS of angular errors if available
    angular_error_cols = [col for col in ['error_roll', 'error_pitch', 'error_yaw'] if col in df.columns]
    if angular_error_cols:
        # Square each error
        for col in angular_error_cols:
            df[f"{col}_squared"] = df[col]**2
        
        # Sum the squares
        df['angular_error_sum_squared'] = df[[f"{col}_squared" for col in angular_error_cols]].sum(axis=1)
        
        # Take the square root to get RMS
        df['angular_error_rms'] = np.sqrt(df['angular_error_sum_squared'])
        
        # Plot RMS angular error
        plt.figure(figsize=(12, 6))
        plt.plot(df['RelativeTime'], df['angular_error_rms'], 'r-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('RMS Angular Error (deg)')
        plt.title('Root Mean Square Angular Error vs Time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'angular_error_rms_plot.png'))
        plt.close()
    
    print(f"All plots have been saved to the '{output_dir}' directory")


if __name__ == "__main__":
    # Simple usage without requiring command line arguments
    csv_file = "test24.csv"  # Put your CSV filename here
    plot_data(csv_file)