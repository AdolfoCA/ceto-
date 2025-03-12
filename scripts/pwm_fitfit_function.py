import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import fsolve
import sys

#sys.path.insert(0, "/home/odroid/catkin_ws/src/rndzvz/data/")

# Define deadzone limits
DEADZONE_LOWER = 1470
DEADZONE_UPPER = 1530

# Load the CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Force (N)'] = data['Force'] * 9.81  # Convert kgf to N if needed
    return data

# Fit two separate polynomials for negative and positive forces, excluding the dead zone
def fit_polynomials(data, degree=4):
    # Split data into negative and positive force regions, excluding the dead zone
    negative_data = data[(data['Force (N)'] < 0) & (data['PWM'] < DEADZONE_LOWER)]
    positive_data = data[(data['Force (N)'] > 0) & (data['PWM'] > DEADZONE_UPPER)]

    # Fit separate polynomials
    coeffs_negative = np.polyfit(negative_data['PWM'], negative_data['Force (N)'], degree)
    coeffs_positive = np.polyfit(positive_data['PWM'], positive_data['Force (N)'], degree)

    return coeffs_negative, coeffs_positive

# Evaluate the polynomial function
def evaluate_polynomial(coeffs, x):
    p = Polynomial(coeffs[::-1])  # Reverse coefficients for Polynomial class
    return p(x)

# Invert the polynomial: Find PWM for a given Force (N)
def force_to_pwm(coeffs_negative, coeffs_positive, desired_force, initial_guess=1500):
    """
    Finds the PWM value corresponding to a given force using the fitted polynomials.
    Separate models are used for positive and negative forces, and dead zone is respected.

    Parameters:
        coeffs_negative (list): Polynomial coefficients for negative forces.
        coeffs_positive (list): Polynomial coefficients for positive forces.
        desired_force (float): The target force (N).
        initial_guess (float): Initial guess for the PWM value.

    Returns:
        float: The estimated PWM value.
    """
    # If the force is in the dead zone range, return neutral PWM
    if -1 <= desired_force <= 1:
        return 1500  # Force within deadzone, keep PWM neutral

    # Select the correct polynomial based on force direction
    if desired_force < 0:
        p = Polynomial(coeffs_negative[::-1])  # Reverse coefficients for Polynomial class
    else:
        p = Polynomial(coeffs_positive[::-1])  # Reverse coefficients for Polynomial class

    # Solve f(PWM) - desired_force = 0 for PWM
    pwm_solution = fsolve(lambda pwm: p(pwm) - desired_force, initial_guess)

    # Ensure the result respects the dead zone
    pwm_value = pwm_solution[0]
    if DEADZONE_LOWER <= pwm_value <= DEADZONE_UPPER:
        return 1500  # If inside deadzone, return neutral PWM

    return int(round(pwm_value))  

# Plot the data and the polynomial fits
def plot_pwm_vs_force(data, coeffs_negative, coeffs_positive):
    plt.figure(figsize=(10, 6))

    # Plot the original data
    plt.scatter(data['PWM'], data['Force (N)'], color='blue', label='Measured Data')

    # Generate x values for the polynomial curves, excluding the dead zone
    x_fit_neg = np.linspace(data['PWM'].min(), DEADZONE_LOWER, 250)  # Negative range
    x_fit_pos = np.linspace(DEADZONE_UPPER, data['PWM'].max(), 250)  # Positive range
    y_fit_neg = evaluate_polynomial(coeffs_negative, x_fit_neg)
    y_fit_pos = evaluate_polynomial(coeffs_positive, x_fit_pos)

    # Plot the polynomial fits
    plt.plot(x_fit_neg, y_fit_neg, color='red', label='Negative Polynomial Fit')
    plt.plot(x_fit_pos, y_fit_pos, color='green', label='Positive Polynomial Fit')

    # Highlight the dead zone region
    plt.axvspan(DEADZONE_LOWER, DEADZONE_UPPER, color='gray', alpha=0.3, label="Dead Zone")

    plt.title('Force vs PWM (Separate Fits for Negative and Positive Forces)')
    plt.xlabel('PWM')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Path to the CSV file
    file_path = "T20016V.csv"

    # Load and process the data
    data = load_data(file_path)

    # Fit separate polynomials for negative and positive forces
    degree = 4
    coeffs_negative, coeffs_positive = fit_polynomials(data, degree)

    # Print polynomial coefficients
    print("Negative Force Polynomial Coefficients:")
    print(coeffs_negative)
    print("Positive Force Polynomial Coefficients:")
    print(coeffs_positive)

    # Plot the data and separate fits
    plot_pwm_vs_force(data, coeffs_negative, coeffs_positive)

    # Example: Convert a force to PWM
    test_forces = [-22, -5, 0, 5, 22]  # Including a deadzone test
    for force in test_forces:
        pwm_value = force_to_pwm(coeffs_negative, coeffs_positive, force)
        print(f"\nEstimated PWM for {force} N: {pwm_value}")
