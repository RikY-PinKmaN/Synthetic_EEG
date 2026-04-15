import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the new dataset
data = """
Subject	Average BL (test)	Highest SO (test)	Highest Best(test)
H1_3x3_117ms	80.0	86.6	65.0
H1_3x3_175ms	87.0	94.4	90.9
H1_5x5_117ms	87.1	63.1	85.5
H1_5x5_175ms	92.7	94.8	96.7
H2_3x3_117ms	63.0	54.7	60.9
H2_3x3_175ms	76.0	55.9	63.4
H2_5x5_117ms	75.5	79.1	78.3
H2_5x5_175ms	80.7	76.6	78.9
H3_3x3_117ms	54.9	51.6	56.2
H3_3x3_175ms	49.5	50.0	48.4
H3_5x5_117ms	62.1	53.9	49.4
H3_5x5_175ms	58.9	48.9	53.9
H4_3x3_117ms	58.8	33.4	53.1
H4_3x3_175ms	72.1	78.8	72.8
H4_5x5_117ms	83.6	70.5	81.2
H4_5x5_175ms	76.7	70.0	76.4
H5_3x3_117ms	87.6	85.3	81.6
H5_3x3_175ms	62.7	70.3	67.2
H5_5x5_117ms	79.3	87.0	79.5
H5_5x5_175ms	80.5	62.7	63.0
H6_3x3_117ms	68.7	55.3	65.6
H6_3x3_175ms	65.8	45.0	61.2
H6_5x5_117ms	77.3	82.5	77.5
H6_5x5_175ms	83.7	86.9	86.7
H7_3x3_117ms	61.9	65.0	66.6
H7_3x3_175ms	65.8	55.9	64.7
H7_5x5_117ms	75.3	83.4	88.4
H7_5x5_175ms	68.7	63.4	66.4
H8_3x3_117ms	76.8	68.4	71.9
H8_3x3_175ms	52.0	34.7	66.6
H8_5x5_117ms	81.8	73.3	82.2
H8_5x5_175ms	87.2	87.2	89.2
H9_3x3_117ms	77.4	76.9	75.6
H9_3x3_175ms	57.8	64.7	64.7
H9_5x5_117ms	70.6	78.4	77.5
H9_5x5_175ms	64.3	65.9	60.9
H10_3x3_117ms	72.1	72.8	77.5
H10_3x3_175ms	89.9	96.9	89.7
H10_5x5_117ms	82.8	92.2	88.4
H10_5x5_175ms	85.7	78.1	82.2
H11_3x3_117ms	73.0	67.5	81.6
H11_3x3_175ms	80.4	76.6	74.1
H11_5x5_117ms	79.4	93.0	86.9
H11_5x5_175ms	79.8	74.7	81.7
H12_3x3_117ms	76.4	70.0	78.1
H12_3x3_175ms	89.2	75.6	84.4
H12_5x5_117ms	70.9	59.2	74.5
H12_5x5_175ms	68.3	62.2	76.9
"""

# Parse the text data (extract only test results)
parsed_data = []
for line in data.strip().split('\n'):
    if line.startswith('H'): 
        parts = line.split()
        subj = parts[0]
        bl_test = float(parts[1])
        so_test = float(parts[2])
        best_test = float(parts[3])
        parsed_data.append([subj, bl_test, so_test, best_test])

# Create the main DataFrame
df = pd.DataFrame(parsed_data, columns=['Subject', 'Baseline', 'Synthetic', 'Best'])

# Subsets for 3x3 and 5x5 matrices
df_3x3 = df[df['Subject'].str.contains('3x3')].copy()
df_5x5 = df[df['Subject'].str.contains('5x5')].copy()

# Dictionary to iterate over for plotting
datasets = {
    'All Data': df,
    '3x3 Grid': df_3x3,
    '5x5 Grid': df_5x5
}

# 2. Setup the Matplotlib figure
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 3, figsize=(18, 17))
fig.suptitle('P300 Test Results Comparison (Baseline vs Synthetic vs Best)', fontsize=20, y=0.99)

# 3. Plotting Loop
for i, (name, data_subset) in enumerate(datasets.items()):
    
    # Dynamics and Stats calculations
    total_samples = len(data_subset)
    imp_synthetic = (data_subset['Synthetic'] > data_subset['Baseline']).sum()
    imp_best = (data_subset['Best'] > data_subset['Baseline']).sum()
    
    # Means for the Box Plot Labels
    mean_bl = data_subset['Baseline'].mean()
    mean_so = data_subset['Synthetic'].mean()
    mean_best = data_subset['Best'].mean()
    
    # Axis limits
    min_val = min(data_subset[['Baseline', 'Synthetic', 'Best']].min()) - 5
    max_val = max(data_subset[['Baseline', 'Synthetic', 'Best']].max()) + 5
    
    # -------------------------------------------------------------
    # Column 1: Scatter Plot - Baseline vs Synthetic
    # -------------------------------------------------------------
    ax1 = axes[i, 0]
    sns.scatterplot(data=data_subset, x='Baseline', y='Synthetic', ax=ax1, s=80, color='royalblue', edgecolor='k')
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x (Equal Perf)')
    ax1.set_title(f'{name}: Baseline vs Synthetic\n(Improved: {imp_synthetic} out of {total_samples})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Baseline Test Accuracy')
    ax1.set_ylabel('Synthetic (SO) Test Accuracy')
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.legend(loc='upper left')

    # -------------------------------------------------------------
    # Column 2: Scatter Plot - Baseline vs Best
    # -------------------------------------------------------------
    ax2 = axes[i, 1]
    sns.scatterplot(data=data_subset, x='Baseline', y='Best', ax=ax2, s=80, color='darkorange', edgecolor='k')
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x (Equal Perf)')
    ax2.set_title(f'{name}: Baseline vs Best\n(Improved: {imp_best} out of {total_samples})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Baseline Test Accuracy')
    ax2.set_ylabel('Best Test Accuracy')
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)
    ax2.legend(loc='upper left')

    # -------------------------------------------------------------
    # Column 3: Boxplots (Baseline vs Synthetic vs Best)
    # -------------------------------------------------------------
    ax3 = axes[i, 2]
    df_melt = data_subset[['Baseline', 'Synthetic', 'Best']].melt(var_name='Model', value_name='Accuracy')
    
    # Draw the boxplot and overlay scatter points
    sns.boxplot(data=df_melt, x='Model', y='Accuracy', ax=ax3, hue='Model', palette='Set2', legend=False, width=0.5)
    sns.stripplot(data=df_melt, x='Model', y='Accuracy', ax=ax3, color=".25", alpha=0.6, jitter=True)
    
    ax3.set_title(f'{name}: Test Distribution Comparison\n(Distributions & Means)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('')
    ax3.set_ylabel('Test Accuracy')
    
    # Apply custom X-axis labels to include the calculated means
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels([
        f"Baseline\n($\mu$={mean_bl:.1f})", 
        f"Synthetic\n($\mu$={mean_so:.1f})", 
        f"Best\n($\mu$={mean_best:.1f})"
    ])

# Adjust padding and layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()