import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

def parse_log_file(file_path):
    """Parses log file, extracts key data: Subject number, Baseline, Synthetic, and Augmented accuracy."""

    def get_real_subject_num(data_dir, folder_subject_num_str):
        folder_subject_num = int(folder_subject_num_str)
        if data_dir == "DATA1_results": return folder_subject_num
        if data_dir == "DATA2_results": return folder_subject_num + 10
        if data_dir == "DATA3_results":
            return folder_subject_num + 20 if folder_subject_num < 9 else 30
        if data_dir == "DATA4_results":
            return folder_subject_num + 30 if folder_subject_num < 4 else folder_subject_num + 31
        if data_dir == "DATA5_results": return folder_subject_num + 40
        return None

    with open(file_path, 'r') as f:
        content = f.read()
    data_dir_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    subdir_match = re.search(r"Subject_(\d+)", os.path.dirname(file_path))
    folder_subject_num = subdir_match.group(1) if subdir_match else None
    subject = None
    if folder_subject_num:
        subject = get_real_subject_num(data_dir_name, folder_subject_num)
    baseline_acc_match = re.search(r'BASELINE ACCURACY \(Real train\+val -> Real test\): ([\d.]+)%', content)
    synth_acc_match = re.search(r'BEST SYNTHETIC train → REAL test accuracy: ([\d.]+)%', content)
    augmented_acc_match = re.search(r'FINAL TEST SET PERFORMANCE OF BEST STRATEGY(?: ---)?\n  Test Accuracy: ([\d.]+)%', content)

    return {
        'Subject': subject,
        'Baseline': float(baseline_acc_match.group(1)) if baseline_acc_match else None,
        'Synthetic': float(synth_acc_match.group(1)) if synth_acc_match else None,
        'Augmented': float(augmented_acc_match.group(1)) if augmented_acc_match else None
    }

def plot_results(df):
    """Generates and saves consolidated plots. Includes consolidated bar plot and box plot."""
    
    # Melt the data to show all three original methods in the plots
    df_melted = df.melt(id_vars=['Subject'],
                        value_vars=['Baseline', 'Synthetic', 'Augmented'],
                        var_name='Accuracy_Type', value_name='Accuracy')

    # --- Consolidated Bar Chart ---
    plt.figure(figsize=(18, 9))
    sns.barplot(data=df_melted, x='Subject', y='Accuracy', hue='Accuracy_Type', palette='viridis')
    plt.title('Accuracy by Subject and Method (Cho2017)', fontsize=20)
    plt.xlabel('Subject', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.xticks(rotation=0)  # Ensure subject labels are readable
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('consolidated_cho2017_barchart.png')
    plt.close()

    # --- Overall Box Plot ---
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=df_melted, x='Accuracy_Type', y='Accuracy', palette='pastel')
    plt.title('Overall Accuracy Distribution (Cho2017)', fontsize=16)
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('all_subjects_cho2017_boxplot.png')
    plt.close()

def perform_statistical_analysis(df):
    """Performs a paired t-test to compare Baseline with the Best Method for Cho2017 dataset."""
    print("\n" + "="*50)
    print(" " * 15 + "Statistical Analysis (Cho2017)")
    print("="*50)

    # Create the 'Best Method' column for statistical comparison only
    df['Best Method'] = df[['Synthetic', 'Augmented']].max(axis=1)

    # --- Paired T-test ---
    baseline_scores = df['Baseline']
    best_method_scores = df['Best Method']
    
    t_stat, p_value = ttest_rel(best_method_scores, baseline_scores)
    
    print("\n--- Paired T-test Results (Baseline vs. Best Method) ---\n")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    print("\n--- Interpretation ---")
    if p_value < 0.05:
        print(f"✓ The difference between 'Best Method' and 'Baseline' is statistically significant (p={p_value:.4f}).")
        if t_stat > 0:
            print("  This suggests the 'Best Method' provides a significant improvement.")
        else:
            print("  This suggests the 'Baseline' is significantly better than the 'Best Method'.")
    else:
        print(f"✗ There is NO statistically significant difference between 'Best Method' and 'Baseline' (p={p_value:.4f}).")
        
    print("\n" + "="*50 + "\n")


if __name__ == '__main__':
    data_dirs = ["DATA1_results", "DATA2_results", "DATA3_results", "DATA4_results", "DATA5_results"]
    log_files = [os.path.join(root, name)
                 for data_dir in data_dirs
                 if os.path.exists(data_dir)
                 for root, _, files in os.walk(data_dir)
                 for name in files
                 if name == 'run_log.txt']

    data = [parse_log_file(log_file) for log_file in log_files]
    df = pd.DataFrame(data)

    df.dropna(inplace=True)
    if 'Subject' in df.columns and not df.empty:
        df['Subject'] = df['Subject'].astype(int)
        df.sort_values(by='Subject', inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("--- Parsed Data (Cho2017) ---")
    print(df.to_string())

    if not df.empty:
        plot_results(df)
        perform_statistical_analysis(df)
        print("\nPlots and statistical analysis generated successfully.")
    else:
        print("\nNo data found to plot or analyze.")