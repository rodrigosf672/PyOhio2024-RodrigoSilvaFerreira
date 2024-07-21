"""
Author: Rodrigo Silva Ferreira
Talk: Signal Processing in Electrochemistry with Python: Applications to the US Opioids Crisis
Date: 28JUL2024 | Event: PyOhio 2024 | Location: Cleveland, OH
Purpose: Demonstrate how the Savitzky-Golay Filter can be helpful to optimize CV signals, with set parameters.
Method: Plot CV data: original data from Fc/Fc+, noisy data, and filtered data using Savitzky-Golay filter
Expectation: Four plots containing original CV data, noisy data, and filtered data with four combinations of window sizes and polynomial orders.
Reference for CV Data: Copley, G.; Gibson, E. Cyclic Voltammetry of a Cobaloxime Catalyst raw data. Newcastle University: 2019 (licensed under CC BY 4.0).
"""

# Import the libraries
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import f_oneway
import pickle

# Load data from CSV file and extract potentials and currents from columns. Set folder for output plots.
file_path = 'CyclicVoltammetryData_RedoxReaction.csv'
data = pd.read_csv(file_path)
E = data['E /V']
I = data['I /uA']
output_folder = 'Script01-Plots'
os.makedirs(output_folder, exist_ok=True)

# Define four different combinations for window sizes and polynomial orders.
combinations = [
    (31, 1),
    (31, 5),
    (101, 1),
    (101, 5),
]

# Define varying levels of noise to the original data and MSLE results to be stored based for combinations.
noise_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
all_msle_results = {combo: [] for combo in combinations}

# Apply random levels of noise, with seed 30 for reproducibility.
np.random.seed(30)
for noise in noise_levels:
    msle_results = []  # Reset for each noise level
    I_noisy = I + np.random.normal(0, max(I) * noise, size=I.shape)  # Calculate max_current within the loop

    # Prepare plots 2x2 with the respective sizes.
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    # Apply the Savitzky-Golay filter and plot them.
    for i, (window_length, poly_order) in enumerate(combinations):
        I_restored = savgol_filter(I_noisy, window_length, poly_order)

        # Define data within specified potential range of the observed peak (-0.5 to 0.2V)
        valid_indices = (E >= -0.5) & (E <= 0.2)
        E_valid = E[valid_indices]
        I_original_valid = I[valid_indices]
        I_restored_valid = I_restored[valid_indices]

        # Filter out non-positive values to avoid runtime warnings
        valid_mask = (I_original_valid > 0) & (I_restored_valid > 0)
        I_original_valid = I_original_valid[valid_mask]
        I_restored_valid = I_restored_valid[valid_mask]

        # Calculate mean squared logarithmic error (MSLE) of the results
        msle = np.mean((np.log(I_original_valid + 1) - np.log(I_restored_valid + 1)) ** 2)
        msle_results.append((window_length, poly_order, msle))
        all_msle_results[(window_length, poly_order)].append(msle)

        # Define parameters for plots
        axs[i].plot(E, I, label='Original CV Data', color='blue', linewidth=2)
        axs[i].plot(E, I_noisy, label='Noisy CV Data', color='red', linewidth=1, alpha=0.5)
        axs[i].plot(E, I_restored, label='Filtered CV Data', color='green', linewidth=2)
        axs[i].set_title(f"Window Size: {window_length}, Polynomial Order: {poly_order}, Noise: {noise}, MSLE: {msle:.2f}")
        axs[i].set_xlabel('Potential (V)')
        axs[i].set_ylabel('Current (µA)')
        axs[i].legend()
        axs[i].invert_xaxis()
        axs[i].set_xticks(np.arange(E.max(), E.min() - 0.1, -0.1))

    # Print results in the terminal
    print(f"Results for Noise Level: {noise}")
    for window_length, poly_order, msle in msle_results:
        print(f"Window Size: {window_length}, Polynomial Order: {poly_order}, Noise: {noise}, MSLE: {msle:.2f}")

    # Create plots and save them in Script01-Plots
    filename = os.path.join(output_folder, f'CV_Signal_Noise{noise}_WithoutHyperOpt.png')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Calculate average MSLE and standard deviation for each model
avg_msle = [np.mean(all_msle_results[combo]) for combo in combinations]
std_dev = [np.std(all_msle_results[combo]) for combo in combinations]

# Perform ANOVA test
anova_result = f_oneway(*[all_msle_results[combo] for combo in combinations])
p_value = anova_result.pvalue

# Plot bar chart of average MSLE values for each model with error bars
models = ['31, 1', '31, 5', '101, 1', '101, 5']
positions = np.arange(len(models))
width = 0.5
fig, ax = plt.subplots()
bars = ax.bar(positions, avg_msle, width, yerr=std_dev, capsize=5, color='skyblue', edgecolor='black')
ax.set_xlabel('Model (Window Size, Polynomial Order)')
ax.set_ylabel('Average MSLE')
ax.set_xticks(positions)
ax.set_xticklabels(models)
ax.set_title(f'Resistance to Noise - Bar Charts with Errors\nOne-Way ANOVA p-value: {p_value:.4f}')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'Resistance_to_Noise_Bar_Charts_with_Errors.jpeg'), dpi=300)

# Store results in a Pickle file for statistical analysis later
pickle_folder = 'Script03-PickleFiles'
os.makedirs(pickle_folder, exist_ok=True)
pickle_file = os.path.join(pickle_folder, 'results_script01.pkl')
results_script01 = {
    'avg_msle': avg_msle,
    'std_dev': std_dev
}
with open(pickle_file, 'wb') as f:
    pickle.dump(results_script01, f)