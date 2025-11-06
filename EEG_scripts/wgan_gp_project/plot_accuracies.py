import re
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel

# --- Definitions for descriptive labels ---
session_details = {
    '1': {'Matrix': '3x3', 'ISI': '175ms', 'Type': 'Quick'},
    '2': {'Matrix': '3x3', 'ISI': '175ms', 'Type': 'Jump'},
    '3': {'Matrix': '3x3', 'ISI': '117ms', 'Type': 'Quick'},
    '4': {'Matrix': '3x3', 'ISI': '117ms', 'Type': 'Jump'},
    '5': {'Matrix': '5x5', 'ISI': '175ms', 'Type': 'Quick'},
    '6': {'Matrix': '5x5', 'ISI': '175ms', 'Type': 'Jump'},
    '7': {'Matrix': '5x5', 'ISI': '117ms', 'Type': 'Quick'},
    '8': {'Matrix': '5x5', 'ISI': '117ms', 'Type': 'Jump'},
}

def generate_descriptive_label(pair_string):
    """Converts a session pair string like "1-2" into a descriptive label."""
    try:
        part1, part2 = pair_string.split('-')
        details1, details2 = session_details[part1], session_details[part2]
        if details1['Matrix'] == details2['Matrix'] and details1['ISI'] == details2['ISI']:
            return f"{details1['Matrix']}, {details1['ISI']} ({details1['Type'][0]}/{details2['Type'][0]})"
        return f"Sess {pair_string}"
    except (KeyError, ValueError):
        return pair_string

# --- Data Parsing ---
data = {}
log_dir_p300 = "P300_results"
if os.path.exists(log_dir_p300):
    for subdir, _, files in os.walk(log_dir_p300):
        if "run_log.txt" in files:
            log_file = os.path.join(subdir, "run_log.txt")
            h_match = re.search(r"(H\d+)_results", subdir)
            s_match = re.search(r"Subject_(\d+-\d+)_results", subdir)
            if h_match and s_match:
                target, session = h_match.group(1), s_match.group(1)
                label = f"{target} Sess {session}"
                if label not in data: data[label] = {}
                with open(log_file, 'r') as f: content = f.read()
                
                baseline_match = re.search(r"Baseline Accuracy \(Train\+Valid -> Test\): (\d+\.?\d*)%", content)
                if baseline_match: data[label]['baseline'] = float(baseline_match.group(1))
                
                synth_only_match = re.search(r"BEST SYNTHETIC-ONLY.*?-> REAL test accuracy: (\d+\.?\d*)%", content)
                if synth_only_match: data[label]['synth_only'] = float(synth_only_match.group(1))

                # --- FINAL CORRECTED PARSING LOGIC ---
                overall_match = re.search(r"BEST OVERALL STRATEGY \((.*?)\).*?-> REAL test accuracy: (\d+\.?\d*)%", content)
                if overall_match:
                    strategy_name = overall_match.group(1)
                    accuracy = float(overall_match.group(2))
                    
                    # If the best strategy was an augmented one, store its accuracy
                    if "Augmented" in strategy_name:
                        if "25%" in strategy_name:
                            data[label]['aug_25'] = accuracy
                        elif "50%" in strategy_name:
                            data[label]['aug_50'] = accuracy
                        elif "100%" in strategy_name:
                            data[label]['aug_100'] = accuracy

# --- Setup ---
metric_names = {
    'baseline': 'Baseline', 'synth_only': 'Synthetic-Only',
    'aug_25': 'Augmented (25%)', 'aug_50': 'Augmented (50%)', 'aug_100': 'Augmented (100%)'
}
methods_for_plot = ['baseline', 'synth_only', 'aug_25', 'aug_50', 'aug_100']

def plot_consolidated_barchart(df):
    """Generates a single consolidated bar chart for all subject-session pairs."""
    plot_methods = [col for col in methods_for_plot if col in df.columns]
    df_melted = df.reset_index().melt(
        id_vars='index', value_vars=plot_methods, var_name='Method', value_name='Accuracy'
    )
    df_melted['Method'] = df_melted['Method'].map(metric_names)

    plt.figure(figsize=(24, 10))
    ax = sns.barplot(data=df_melted, x='index', y='Accuracy', hue='Method', palette='viridis')
    
    original_labels = sorted(df.index.unique())
    session_parts = [lbl.split(' Sess ')[1] for lbl in original_labels]
    descriptive_labels = [f"{orig.split(' ')[0]} ({generate_descriptive_label(sess)})" for orig, sess in zip(original_labels, session_parts)]
    ax.set_xticks(range(len(descriptive_labels)))
    ax.set_xticklabels(descriptive_labels, rotation=45, ha="right")

    plt.title('Model Accuracies by Subject and Session', fontsize=20)
    plt.xlabel('Subject and Session Condition', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Method')
    plt.tight_layout()
    plt.savefig('consolidated_accuracies_barchart.png')
    plt.close()
    print("\nConsolidated bar chart saved as consolidated_accuracies_barchart.png")

def plot_consolidated_boxplot(df):
    """Generates a consolidated box plot of all methods."""
    plot_methods = [col for col in methods_for_plot if col in df.columns]
    plot_data = [df[col].dropna().tolist() for col in plot_methods if not df[col].dropna().empty]
    plot_labels = [metric_names.get(col) for col in plot_methods if not df[col].dropna().empty]
    
    if plot_data:
        plt.figure(figsize=(12, 8))
        plt.boxplot(plot_data, tick_labels=plot_labels)
        plt.title('Overall Model Accuracy Distribution', fontsize=16)
        plt.ylabel('Accuracy (%)')
        plt.ylim(20, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig('p300_accuracies_boxplot.png')
        plt.close()
        print("Consolidated boxplot saved as p300_accuracies_boxplot.png")

def perform_statistical_analysis(df):
    """Performs a paired t-test comparing Baseline to the Best Method."""
    gan_methods = ['synth_only', 'aug_25', 'aug_50', 'aug_100']
    existing_gan_methods = [col for col in gan_methods if col in df.columns]
    
    if not existing_gan_methods:
        print("\nNo synthetic or augmented data found for statistical analysis.")
        return

    df['Best Method'] = df[existing_gan_methods].max(axis=1)
    
    print("\n" + "="*50 + "\n" + " " * 15 + "Statistical Analysis" + "\n" + "="*50)
    comparison_df = df[['baseline', 'Best Method']].dropna()

    if len(comparison_df) > 1:
        t_stat, p_value = ttest_rel(comparison_df['Best Method'], comparison_df['baseline'])
        print("\n--- Paired T-test Results (Baseline vs. Best Method) ---\n")
        print(f"Comparison: Baseline vs. Best Method (highest of available GAN methods)")
        print(f"Number of paired samples: {len(comparison_df)}")
        print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

        print("\n--- Interpretation ---")
        if p_value < 0.05:
            if t_stat > 0:
                print(f"✓ The 'Best Method' provides a statistically significant improvement over the 'Baseline'.")
            else:
                print(f"✓ The 'Baseline' is statistically significantly better than the 'Best Method'.")
        else:
            print(f"✗ There is NO statistically significant difference between 'Best Method' and 'Baseline'.")
    else:
        print("\nNot enough paired data to perform a statistical test.")
    print("\n" + "="*50 + "\n")

if not data:
    print("No data found to plot or analyze.")
else:
    df = pd.DataFrame.from_dict(data, orient='index')
    df.sort_index(inplace=True)

    print("--- Parsed Data ---")
    print(df.to_string())

    plot_consolidated_barchart(df)
    plot_consolidated_boxplot(df)
    perform_statistical_analysis(df)