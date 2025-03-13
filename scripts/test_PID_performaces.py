#!/usr/bin/env python3
"""
Script to test PID controller performance with and without homomorphic encryption.
This script helps identify the computational overhead of using Paillier encryption
for PID control loops in ROV applications.
"""

import time
import numpy as np
import rospy
from phe import paillier
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Configuration parameters
USE_ENCRYPTION = True
PUBLIC_KEY_SIZE = 512  # Try different sizes: 256, 512, 1024, 2048
DETAILED_TIMING = True  # Set to True for detailed timing of each PID operation
SAVE_GRAPHS = True  # Save graphs to disk
OUTPUT_DIR = "pid_performance_results"  # Directory to save results

# Create output directory if needed
if SAVE_GRAPHS and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Generate keypair with specified key size if using encryption
if USE_ENCRYPTION:
    print(f"Generating {PUBLIC_KEY_SIZE}-bit keypair...")
    start_time = time.time()
    P, S = paillier.generate_paillier_keypair(n_length=PUBLIC_KEY_SIZE)
    key_gen_time = time.time() - start_time
    print(f"Keypair generated in {key_gen_time:.2f} seconds")
else:
    P, S = None, None

class PID:
    """Full PID controller implementation with homomorphic encryption support."""
    def __init__(self, kp, ki, kd, setpoint=None, integral_limit=4):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit  # Limit for integral term
        
        # Timing statistics
        self.times = {
            'error': [],
            'integral': [],
            'derivative': [],
            'p_term': [],
            'i_term': [],
            'd_term': [],
            'sum': [],
            'total': []
        }
        
        # Handle encryption properly for initial values
        if USE_ENCRYPTION:
            if isinstance(setpoint, paillier.EncryptedNumber):
                self.setpoint = setpoint
            elif setpoint is None:
                self.setpoint = P.encrypt(0)
            else:
                self.setpoint = P.encrypt(setpoint)
                
            # Initialize with encrypted zeros
            self.prev_error = P.encrypt(0)
            self.integral = P.encrypt(0)
        else:
            self.setpoint = 0 if setpoint is None else setpoint
            self.prev_error = 0
            self.integral = 0
            
    def update(self, measurement, setpoint=None):
        """Calculate the PID control output with support for encrypted values."""
        total_start = time.time()
        
        # Setpoint check
        if setpoint is None:
            setpoint = self.setpoint
        
        # Error calculation
        error_start = time.time()
        error = setpoint - measurement
        error_time = time.time() - error_start
        self.times['error'].append(error_time)
        
        # Integral calculation
        integral_start = time.time()
        self.integral = self.integral + error
        integral_time = time.time() - integral_start
        self.times['integral'].append(integral_time)
        
        # Limit integral term if not encrypted
        if not USE_ENCRYPTION and self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        # Derivative calculation
        derivative_start = time.time()
        derivative = error - self.prev_error
        self.prev_error = error
        derivative_time = time.time() - derivative_start
        self.times['derivative'].append(derivative_time)
        
        # Term calculations
        p_term_start = time.time()
        p_term = self.kp * error
        p_term_time = time.time() - p_term_start
        self.times['p_term'].append(p_term_time)
        
        i_term_start = time.time()
        i_term = self.ki * self.integral
        i_term_time = time.time() - i_term_start
        self.times['i_term'].append(i_term_time)
        
        d_term_start = time.time()
        d_term = self.kd * derivative
        d_term_time = time.time() - d_term_start
        self.times['d_term'].append(d_term_time)
        
        # Sum calculation
        sum_start = time.time()
        result = p_term + i_term + d_term
        sum_time = time.time() - sum_start
        self.times['sum'].append(sum_time)
        
        # Total time
        total_time = time.time() - total_start
        self.times['total'].append(total_time)
        
        return result
    
    def print_stats(self):
        """Print timing statistics."""
        print("\nPID Performance Statistics:")
        print("-" * 50)
        
        for key, times in self.times.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                print(f"{key.capitalize()} Operation: "
                      f"Avg={avg_time*1000:.3f}ms, "
                      f"Min={min_time*1000:.3f}ms, "
                      f"Max={max_time*1000:.3f}ms")
        
        print("-" * 50)
        print(f"Total calls: {len(self.times['total'])}")
        if self.times['total']:
            total_avg = sum(self.times['total']) / len(self.times['total'])
            print(f"Average time per update: {total_avg*1000:.3f}ms")
            print(f"Maximum control frequency: {1/total_avg:.2f}Hz")
        print("-" * 50)
        
        return {key: sum(times)/len(times) if times else 0 for key, times in self.times.items()}


def plot_pid_timings(pid_results, title, filename=None):
    """Create a bar chart of PID operation timings."""
    labels = ['Error', 'Integral', 'Derivative', 'P Term', 'I Term', 'D Term', 'Sum', 'Total']
    values = [
        pid_results['error'] * 1000,  # Convert to ms
        pid_results['integral'] * 1000,
        pid_results['derivative'] * 1000,
        pid_results['p_term'] * 1000,
        pid_results['i_term'] * 1000,
        pid_results['d_term'] * 1000,
        pid_results['sum'] * 1000,
        pid_results['total'] * 1000
    ]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, values, color='skyblue')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                 f'{height:.2f}ms', ha='center', va='bottom')
    
    plt.title(title)
    plt.ylabel('Average Time (ms)')
    plt.grid(axis='y', alpha=0.3)
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if filename and SAVE_GRAPHS:
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
    
    plt.show()


def plot_operation_comparison(normal_results, encrypted_results, title, filename=None):
    """Create a bar chart comparing normal vs encrypted operations."""
    operations = ['Error', 'Integral', 'Derivative', 'P Term', 'I Term', 'D Term', 'Sum', 'Total']
    
    normal_values = [
        normal_results['error'] * 1000,
        normal_results['integral'] * 1000,
        normal_results['derivative'] * 1000,
        normal_results['p_term'] * 1000,
        normal_results['i_term'] * 1000,
        normal_results['d_term'] * 1000,
        normal_results['sum'] * 1000,
        normal_results['total'] * 1000
    ]
    
    encrypted_values = [
        encrypted_results['error'] * 1000,
        encrypted_results['integral'] * 1000,
        encrypted_results['derivative'] * 1000,
        encrypted_results['p_term'] * 1000,
        encrypted_results['i_term'] * 1000,
        encrypted_results['d_term'] * 1000,
        encrypted_results['sum'] * 1000,
        encrypted_results['total'] * 1000
    ]
    
    x = np.arange(len(operations))  # Label locations
    width = 0.35  # Width of the bars
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, normal_values, width, label='Normal', color='skyblue')
    rects2 = ax.bar(x + width/2, encrypted_values, width, label=f'Encrypted ({PUBLIC_KEY_SIZE}-bit)', color='lightcoral')
    
    # Add labels and customize
    ax.set_ylabel('Time (ms)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(operations, rotation=45)
    ax.legend()
    
    # Add speedup numbers
    for i, (n, e) in enumerate(zip(normal_values, encrypted_values)):
        if n > 0:  # Avoid division by zero
            speedup = e / n
            ax.text(i, max(n, e) + 0.1, f'{speedup:.1f}x', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Add values on bars for better readability
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if filename and SAVE_GRAPHS:
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
    
    plt.show()


def plot_update_times(time_series, labels, title, filename=None):
    """Create a line chart of update times over iterations."""
    plt.figure(figsize=(14, 8))
    
    for times, label in zip(time_series, labels):
        # Convert to milliseconds
        ms_times = [t * 1000 for t in times]
        plt.plot(range(len(ms_times)), ms_times, label=label, alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Update Time (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add horizontal lines for control frequencies
    frequencies = [10, 20, 50, 100]
    for freq in frequencies:
        ms_time = 1000 / freq
        if ms_time <= max([max(t * 1000) for t in time_series]):
            plt.axhline(y=ms_time, color='r', linestyle='--', alpha=0.5)
            plt.text(0, ms_time, f'{freq} Hz', va='bottom')
    
    plt.tight_layout()
    
    if filename and SAVE_GRAPHS:
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
    
    plt.show()


def plot_key_size_comparison(results, title, filename=None):
    """Create a chart comparing performance across key sizes."""
    key_sizes = list(results.keys())
    operations = ['Encrypt', 'Decrypt', 'Add', 'Multiply']
    
    # Extract data
    data = {
        'encrypt': [results[size]['encrypt'] * 1000 for size in key_sizes],
        'decrypt': [results[size]['decrypt'] * 1000 for size in key_sizes],
        'add': [results[size]['add'] * 1000 for size in key_sizes],
        'mul': [results[size]['mul'] * 1000 for size in key_sizes]
    }
    
    # Plot bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(key_sizes))
    width = 0.2
    multiplier = 0
    
    for operation, times in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, times, width, label=operation.capitalize())
        multiplier += 1
    
    # Add labels and customize
    ax.set_title(title)
    ax.set_xlabel('Key Size (bits)')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{size}-bit" for size in key_sizes])
    ax.legend(loc='best')
    plt.grid(axis='y', alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization
    
    # Add value annotations
    for i, v in enumerate(data['encrypt']):
        plt.text(i - 0.3, v * 1.1, f"{v:.2f}", ha='center', fontsize=8)
    for i, v in enumerate(data['decrypt']):
        plt.text(i - 0.1, v * 1.1, f"{v:.2f}", ha='center', fontsize=8)
    for i, v in enumerate(data['add']):
        plt.text(i + 0.1, v * 1.1, f"{v:.2f}", ha='center', fontsize=8)
    for i, v in enumerate(data['mul']):
        plt.text(i + 0.3, v * 1.1, f"{v:.2f}", ha='center', fontsize=8)
    
    plt.tight_layout()
    
    if filename and SAVE_GRAPHS:
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
    
    plt.show()
    
    # Also create a line plot for clearer trend visualization
    plt.figure(figsize=(12, 8))
    
    for operation, times in data.items():
        plt.plot(key_sizes, times, marker='o', label=operation.capitalize())
    
    plt.title(f"{title} (Trend)")
    plt.xlabel('Key Size (bits)')
    plt.ylabel('Time (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')
    
    if filename and SAVE_GRAPHS:
        trend_filename = filename.replace('.png', '_trend.png')
        plt.savefig(os.path.join(OUTPUT_DIR, trend_filename))
    
    plt.show()


def test_setpoint_change(pid, measurements, setpoints):
    """Test PID performance with changing setpoints and measurements."""
    print(f"\nTesting PID with {'encrypted' if USE_ENCRYPTION else 'normal'} values...")
    
    results = []
    start_time = time.time()
    
    for i, (m, s) in enumerate(zip(measurements, setpoints)):
        if i % 10 == 0:
            print(f"Running iteration {i+1}/{len(measurements)}...")
        
        # Encrypt values if needed
        if USE_ENCRYPTION:
            m_enc = P.encrypt(m)
            s_enc = P.encrypt(s)
            result = pid.update(m_enc, s_enc)
            # Decrypt for validation
            results.append(S.decrypt(result))
        else:
            result = pid.update(m, s)
            results.append(result)
    
    total_time = time.time() - start_time
    print(f"Completed {len(measurements)} iterations in {total_time:.3f} seconds")
    print(f"Average time per iteration: {(total_time/len(measurements))*1000:.3f}ms")
    
    return results


def test_pid_performance():
    """Comprehensive test function to measure PID performance with and without encryption."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("TESTING PID CONTROLLER PERFORMANCE")
    print("=" * 60)
    
    # Test parameters
    kp, ki, kd = 0.1, 0.01, 0.05
    iterations = 100
    warmup_iterations = 5
    
    # Generate test data
    measurements = [1 + 0.1*i for i in range(iterations)]
    setpoints = [2 + 0.05*i for i in range(iterations)]
    
    # Define test cases
    tests = [
        {"name": "Normal PID", "use_encryption": False},
        {"name": f"Encrypted PID ({PUBLIC_KEY_SIZE}-bit)", "use_encryption": True}
    ]
    
    results = {}
    pid_stats = {}
    time_series = []
    labels = []
    
    # Run tests
    for test in tests:
        print("\n" + "=" * 60)
        print(f"TEST CASE: {test['name']}")
        print("=" * 60)
        
        global USE_ENCRYPTION
        USE_ENCRYPTION = test['use_encryption']
        
        # Initialize PID controller
        pid = PID(kp=kp, ki=ki, kd=kd)
        
        # Warm-up
        print("Performing warm-up...")
        warmup_data = measurements[:warmup_iterations]
        warmup_setpoints = setpoints[:warmup_iterations]
        test_setpoint_change(pid, warmup_data, warmup_setpoints)
        
        # Reset timing stats after warm-up
        pid.times = {key: [] for key in pid.times}
        
        # Run actual test
        print("Starting performance test...")
        results[test['name']] = test_setpoint_change(pid, measurements, setpoints)
        
        # Store timing stats
        stats = pid.print_stats()
        pid_stats[test['name']] = stats
        
        # Store time series for iteration plot
        time_series.append(pid.times['total'])
        labels.append(test['name'])
        
        # Plot individual PID timings
        graph_title = f"PID Operation Timings - {test['name']}"
        filename = f"pid_timings_{test['name'].replace(' ', '_').lower()}_{timestamp}.png"
        plot_pid_timings(stats, graph_title, filename)
    
    # Plot comparison between normal and encrypted
    if len(pid_stats) > 1:
        comparison_title = f"PID Performance Comparison - Normal vs. Encrypted ({PUBLIC_KEY_SIZE}-bit)"
        comparison_filename = f"pid_comparison_{PUBLIC_KEY_SIZE}bit_{timestamp}.png"
        plot_operation_comparison(
            pid_stats["Normal PID"],
            pid_stats[f"Encrypted PID ({PUBLIC_KEY_SIZE}-bit)"],
            comparison_title,
            comparison_filename
        )
        
        # Plot update times over iterations
        timeseries_title = "PID Update Times Over Iterations"
        timeseries_filename = f"pid_update_times_{timestamp}.png"
        plot_update_times(time_series, labels, timeseries_title, timeseries_filename)
    
    # Compare results for validation
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("RESULT VALIDATION")
        print("=" * 60)
        
        normal_results = results["Normal PID"]
        encrypted_results = results[f"Encrypted PID ({PUBLIC_KEY_SIZE}-bit)"]
        
        # Verify first few values
        print("\nSample results comparison (first 5 values):")
        for i in range(min(5, len(normal_results))):
            print(f"Iteration {i}: Normal={normal_results[i]:.6f}, "
                  f"Encrypted={encrypted_results[i]:.6f}, "
                  f"Diff={abs(normal_results[i]-encrypted_results[i]):.6f}")
        
        # Calculate error statistics
        diffs = [abs(n-e) for n, e in zip(normal_results, encrypted_results)]
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        
        print(f"\nAverage difference: {avg_diff:.6f}")
        print(f"Maximum difference: {max_diff:.6f}")
        
        # Plot difference histogram
        plt.figure(figsize=(10, 6))
        plt.hist(diffs, bins=20, alpha=0.7, color='skyblue')
        plt.axvline(avg_diff, color='r', linestyle='--', label=f'Avg diff: {avg_diff:.6f}')
        plt.title("Difference Between Normal and Encrypted Results")
        plt.xlabel("Absolute Difference")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        
        diff_filename = f"result_differences_{PUBLIC_KEY_SIZE}bit_{timestamp}.png"
        if SAVE_GRAPHS:
            plt.savefig(os.path.join(OUTPUT_DIR, diff_filename))
        plt.show()
        
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST COMPLETE")
    print("=" * 60)
    
    return pid_stats


def test_key_size_impact():
    """Test the impact of key size on encryption performance."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("TESTING KEY SIZE IMPACT ON PERFORMANCE")
    print("=" * 60)
    
    key_sizes = [256, 512, 1024, 2048]
    results = {}
    
    for size in key_sizes:
        print(f"\nTesting {size}-bit key size:")
        
        # Generate keypair
        start_time = time.time()
        p, s = paillier.generate_paillier_keypair(n_length=size)
        key_gen_time = time.time() - start_time
        print(f"Keypair generated in {key_gen_time:.3f} seconds")
        
        # Test operations
        value = 10.5
        
        # Test encryption
        start_time = time.time()
        iterations = 10
        for _ in range(iterations):
            encrypted = p.encrypt(value)
        encrypt_time = (time.time() - start_time) / iterations
        
        # Test decryption
        start_time = time.time()
        for _ in range(iterations):
            decrypted = s.decrypt(encrypted)
        decrypt_time = (time.time() - start_time) / iterations
        
        # Test addition
        e1 = p.encrypt(5.0)
        e2 = p.encrypt(3.0)
        start_time = time.time()
        for _ in range(iterations):
            result = e1 + e2
        add_time = (time.time() - start_time) / iterations
        
        # Test multiplication
        e1 = p.encrypt(5.0)
        start_time = time.time()
        for _ in range(iterations):
            result = e1 * 2.5
        mul_time = (time.time() - start_time) / iterations
        
        # Record results
        results[size] = {
            "key_gen": key_gen_time,
            "encrypt": encrypt_time,
            "decrypt": decrypt_time,
            "add": add_time,
            "mul": mul_time
        }
        
        # Print results
        print(f"Encryption: {encrypt_time*1000:.3f}ms per operation")
        print(f"Decryption: {decrypt_time*1000:.3f}ms per operation")
        print(f"Addition: {add_time*1000:.3f}ms per operation")
        print(f"Multiplication: {mul_time*1000:.3f}ms per operation")
    
    # Compare results
    print("\n" + "=" * 60)
    print("KEY SIZE COMPARISON")
    print("=" * 60)
    
    operations = ["encrypt", "decrypt", "add", "mul"]
    
    print(f"{'Operation':<15} " + " ".join(f"{size:>8}-bit" for size in key_sizes))
    print("-" * (15 + 9 * len(key_sizes)))
    
    for op in operations:
        print(f"{op:<15} " + " ".join(f"{results[size][op]*1000:>8.2f}ms" for size in key_sizes))
    
    # Calculate scaling factors
    base_size = key_sizes[0]
    print("\nScaling factors (relative to smallest key size):")
    for op in operations:
        base = results[base_size][op]
        print(f"{op:<15} " + " ".join(f"{results[size][op]/base:>8.2f}x" for size in key_sizes))
    
    # Plot key size comparison
    plot_title = "Effect of Key Size on Homomorphic Operation Performance"
    plot_filename = f"key_size_comparison_{timestamp}.png"
    plot_key_size_comparison(results, plot_title, plot_filename)
    
    # Create predicted control frequency plot
    plt.figure(figsize=(10, 6))
    
    # Calculate maximum control frequencies
    frequencies = []
    for size in key_sizes:
        # Estimate time for a full PID update (all ops + overhead)
        est_time = (
            results[size]["encrypt"] * 3 +  # Error, setpoint, measurement
            results[size]["add"] * 3 +      # Integral, P+I, P+I+D
            results[size]["mul"] * 3 +      # P term, I term, D term
            results[size]["decrypt"]        # Final result
        )
        max_freq = 1 / est_time
        frequencies.append(max_freq)
    
    plt.bar(range(len(key_sizes)), frequencies, width=0.6, color='skyblue')
    
    # Add value labels on bars
    for i, freq in enumerate(frequencies):
        plt.text(i, freq + 0.5, f"{freq:.1f} Hz", ha='center')
    
    # Add reference line for 10Hz
    plt.axhline(y=10, color='r', linestyle='--', label='10 Hz (typical control)')
    
    plt.title("Estimated Maximum Control Frequency by Key Size")
    plt.xlabel("Key Size (bits)")
    plt.xticks(range(len(key_sizes)), [f"{size}-bit" for size in key_sizes])
    plt.ylabel("Maximum Frequency (Hz)")
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    freq_filename = f"estimated_control_frequency_{timestamp}.png"
    if SAVE_GRAPHS:
        plt.savefig(os.path.join(OUTPUT_DIR, freq_filename))
    plt.show()
    
    return results


if __name__ == "__main__":
    print("PID Performance Testing with Visualization")
    print("=" * 60)
    
    if SAVE_GRAPHS:
        print(f"Graphs will be saved to: {os.path.abspath(OUTPUT_DIR)}")
    
    # Run the PID performance test
    pid_stats = test_pid_performance()
    
    # Optionally test key size impact
    print("\nWould you like to test the impact of key size on performance? (y/n)")
    try:
        choice = input().strip().lower()
        if choice == 'y':
            key_size_results = test_key_size_impact()
    except:
        # Handle the case where running in a non-interactive environment
        print("Skipping key size impact test (non-interactive mode)")
    
    print("\nTests complete.")