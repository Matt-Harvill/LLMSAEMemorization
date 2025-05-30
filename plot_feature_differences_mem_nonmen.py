import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Directory containing the CSV files
data_dir = 'aggregate_feature_comparisons'

# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(data_dir, 'feature_counts_layer_*.csv'))

# Read and concatenate all CSV files
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(combined_df['Percent_Difference'], bins=30, edgecolor='black')
plt.xlabel('Percent Difference (100% means only memorized XOR non-memorized sequences have this feature)')
plt.ylabel('Frequency')
plt.title('Percent Differences for Most Polarizing SAE Features across Memorized vs Non-Memorized Sequences')
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('percent_differences_histogram.png')
plt.close()

# Print some basic statistics
print("\nSummary Statistics:")
print(combined_df['Percent_Difference'].describe()) 