"""
Author: Rodrigo Silva Ferreira
Talk: Signal Processing in Electrochemistry with Python: Applications to the US Opioids Crisis
Date: 28JUL2024 | Event: PyOhio 2024 | Location: Cleveland, OH
Purpose: Compare results between Script01 and Script02, to determine the impact of using HyperOpt on selecting parameters for SavGol Filter.
Expectation: Unsure... pure curiosity! Does using HyperOpt help achieve better results? If not, are differences statistically significant?
Reference for CV Data: Copley, G.; Gibson, E. Cyclic Voltammetry of a Cobaloxime Catalyst raw data. Newcastle University: 2019 (licensed under CC BY 4.0).
"""

# Import libraries
import pickle
import numpy as np
from scipy.stats import ttest_ind
import os

# Load results from Script01 and Script02
folder_path = 'Script03-PickleFiles'
file_path_script01 = os.path.join(folder_path, 'results_script01.pkl')
with open(file_path_script01, 'rb') as f:
    results_script01 = pickle.load(f)
file_path_script02 = os.path.join(folder_path, 'results_script02.pkl')
with open(file_path_script02, 'rb') as f:
    results_script02 = pickle.load(f)

# Calculate average of the average MSLEs and standard deviations for Script01 and Script02
avg_msle_script01 = np.mean(results_script01['avg_msle'])
std_dev_script01 = np.mean(results_script01['std_dev'])
avg_msle_script02 = np.mean(results_script02['avg_msle'])
std_dev_script02 = np.mean(results_script02['std_dev'])
print(f"Script01 - Average MSLE: {avg_msle_script01:.4f}, Standard Deviation: {std_dev_script01:.4f}")
print(f"Script02 - Average MSLE: {avg_msle_script02:.4f}, Standard Deviation: {std_dev_script02:.4f}")

# Perform two-sample t-test between averages
t_stat, p_value = ttest_ind(results_script01['avg_msle'], results_script02['avg_msle'])
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

# Determine if the difference is statistically significant based on p-value.
alpha = 0.05
if p_value < alpha:
    print("There is a statistically significant difference between the two sets of average MSLEs.")
else:
    print("There is NO statistically significant difference between the two sets of average MSLEs.")