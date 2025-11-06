import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

def parse_log_file(file_path):
    """
    Parses a log file to extract experimental parameters and results.
    Information is extracted from both the log file content and its directory path.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract information from the directory name
    dir_name = os.path.basename(os.path.dirname(file_path))
    
    # Subject from directory name, e.g., "Subject_1_..." -> 1
    subject_match = re.search(r'Subject_(\d+)', dir_name)
    subject = int(subject_match.group(1)) if subject_match else None

    # Training session from directory name, e.g., "..._D1T_..." -> 1
    train_session_match = re.search(r'D(\d)T', dir_name)
    train_session = int(train_session_match.group(1)) if train_session_match else None

    # Determine the condition based on the training session
    if train_session == 1:
        condition = 'Train S1 / Test S2'
    elif train_session == 2:
        condition = 'Train S2 / Test S1'
    else:
        condition = 'Unknown'

    # Number of trials from directory name, e.g., "..._25trials" -> 25
    trials_match = re.search(r'(\d+)trials', dir_name)
    trials = int(trials_match.group(1)) if trials_match else None

    # Extract accuracy values from the log file content
    baseline_acc_match = re.search(r'REAL train → REAL test accuracy \(Baseline\): ([\d.]+)%', content)
    baseline_acc = float(baseline_acc_match.group(1)) if baseline_acc_match else None

    synth_acc_match = re.search(r'BEST SYNTHETIC train → REAL test accuracy: ([\d.]+)%', content)
    synth_acc = float(synth_acc_match.group(1)) if synth_acc_match else None

    augmented_acc_match = re.search(r'FINAL TEST SET PERFORMANCE OF BEST STRATEGY(?: ---)?\n  Test Accuracy: ([\d.]+)%', content)
    augmented_acc = float(augmented_acc_match.group(1)) if augmented_acc_match else None

    return {
        'Subject': subject,
        'Condition': condition,
        'Trials': trials,
        'Baseline': baseline_acc,
        'Synthetic': synth_acc,
        'Augmented': augmented_acc
    }

def plot_results(df):
    """
    Generates and saves consolidated plots to visualize the results for all three methods.
    """
    # Melt the dataframe to long format for easier plotting with seaborn, keeping all three methods
    df_melted = df.melt(id_vars=['Subject', 'Condition', 'Trials'], 
                        value_vars=['Baseline', 'Synthetic', 'Augmented'],
                        var_name='Accuracy Type', value_name='Accuracy')

    # --- Trial-wise Consolidated Bar Charts ---
    # For each trial number, create faceted bar charts (one for each training condition)
    for trial in sorted(df['Trials'].unique()):
        trial_df = df_melted[df_melted['Trials'] == trial]
        
        g = sns.catplot(
            data=trial_df,
            x='Subject', y='Accuracy', hue='Accuracy Type',
            col='Condition',  # Creates separate plots for each condition
            kind='bar',
            palette='muted',
            height=6, aspect=1.1
        )
        
        g.fig.suptitle(f'Accuracy for {trial} Trials (Grouped by Condition)', y=1.03, fontsize=16)
        g.set_axis_labels("Subject", "Accuracy (%)")
        g.set_titles("{col_name}")
        g.despine(left=True)
        g.set(ylim=(0, 100))
        
        plt.savefig(f'{trial}_trials_accuracy_barchart.png')
        plt.close()

    # --- Overall Box Plot ---
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=df_melted, x='Accuracy Type', y='Accuracy', palette='pastel')
    plt.title('Overall Accuracy Distribution', fontsize=16)
    plt.xlabel('Accuracy Type', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('all_results_boxplot.png')
    plt.close()

def perform_statistical_analysis(df):
    """
    Performs a paired t-test to compare Baseline accuracy with the Best Method (higher of Synthetic/Augmented).
    """
    print("\n" + "="*50)
    print(" " * 15 + "Statistical Analysis")
    print("="*50)

    # Create the 'Best Method' column for statistical comparison only
    df['Best Method'] = df[['Synthetic', 'Augmented']].max(axis=1)

    # --- Paired T-test ---
    # Compare the 'Baseline' with the 'Best Method'
    baseline_scores = df['Baseline']
    best_method_scores = df['Best Method']
    
    # Perform the paired t-test
    t_stat, p_value = ttest_rel(best_method_scores, baseline_scores)
    
    print("\n--- Paired T-test Results (Baseline vs. Best Method) ---\n")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    print("\n--- Interpretation ---")
    if p_value < 0.05:
        print(f"✓ The difference between the 'Best Method' and 'Baseline' is statistically significant (p={p_value:.4f}).")
        if t_stat > 0:
            print("  The 'Best Method' provides a statistically significant improvement over the 'Baseline'.")
        else:
            print("  The 'Baseline' is statistically significantly better than the 'Best Method'.")
    else:
        print(f"✗ There is NO statistically significant difference between the 'Best Method' and 'Baseline' (p={p_value:.4f}).")
        
    print("\n" + "="*50 + "\n")


if __name__ == '__main__':
    # Find all 'run_log.txt' files within the 'ResUlTS' directory
    log_files = [os.path.join(root, name)
                 for root, dirs, files in os.walk('ResUlTS')
                 for name in files
                 if name == 'run_log.txt']
    
    # Parse each log file and store the data
    data = [parse_log_file(log_file) for log_file in log_files]
    df = pd.DataFrame(data)
    
    # Drop rows where any data might be missing
    df.dropna(inplace=True)

    # Sort values for consistent plotting
    df.sort_values(by=['Subject', 'Condition', 'Trials'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print("Parsed Data:")
    print(df)
    
    # Generate and save all plots and analysis
    if not df.empty:
        plot_results(df)
        perform_statistical_analysis(df)
        print("\nPlots and statistical analysis generated successfully.")
    else:
        print("\nNo data to plot or analyze.")