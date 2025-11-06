'''
This script combines parsing logic to read log files from P300, BCI2a, and 
Cho2017 experiments. It then generates a series of scatter plots to compare 
baseline, synthetic-only, and best-augmented data performance for each specific
dataset and experimental condition.
'''

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_p300_logs(root_dir='P300_results'):
    """Parses logs for the P300 dataset."""
    data = []
    if not os.path.exists(root_dir):
        return data

    for subdir, _, files in os.walk(root_dir):
        if "run_log.txt" in files:
            log_file = os.path.join(subdir, "run_log.txt")
            h_match = re.search(r"(H\d+)_results", subdir)
            s_match = re.search(r"Subject_(\d+-\d+)_results", subdir)
            
            if h_match and s_match:
                session_group = h_match.group(1)
                subject_pair = s_match.group(1)
                
                with open(log_file, 'r') as f:
                    content = f.read()

                baseline_match = re.search(r"Baseline Accuracy \(Train\+Valid -> Test\): (\d+\.?\d*)%", content)
                synth_only_match = re.search(r"BEST SYNTHETIC-ONLY.*?-> REAL test accuracy: (\d+\.?\d*)%", content)
                
                best_aug_accuracy = None
                overall_match = re.search(r"BEST OVERALL STRATEGY \((.*?)\).*?-> REAL test accuracy: (\d+\.?\d*)%", content)
                if overall_match:
                    strategy_name = overall_match.group(1)
                    accuracy = float(overall_match.group(2))
                    if "Augmented" in strategy_name:
                        best_aug_accuracy = accuracy

                if baseline_match and synth_only_match and best_aug_accuracy is not None:
                    data.append({
                        'Dataset': 'P300',
                        'Subject': f"{session_group}-S{subject_pair}",
                        'Baseline': float(baseline_match.group(1)),
                        'Synthetic': float(synth_only_match.group(1)),
                        'Best_Augmented': best_aug_accuracy
                    })
    return data

def parse_bci2a_logs(root_dir='ResUlTS'):
    """Parses logs for the BCI Competition IV-2a dataset."""
    data = []
    if not os.path.exists(root_dir):
        return data
        
    log_files = [os.path.join(root, name)
                 for root, _, files in os.walk(root_dir)
                 for name in files if name == 'run_log.txt']

    for log_file in log_files:
        with open(log_file, 'r') as f:
            content = f.read()

        dir_name = os.path.basename(os.path.dirname(log_file))
        
        subject_match = re.search(r'Subject_(\d+)', dir_name)
        train_session_match = re.search(r'D(\d)T', dir_name)
        trials_match = re.search(r'(\d+)trials', dir_name)

        if not (subject_match and train_session_match and trials_match):
            continue

        subject = int(subject_match.group(1))
        train_session = int(train_session_match.group(1))
        trials = int(trials_match.group(1))

        baseline_acc_match = re.search(r'REAL train \u2192 REAL test accuracy \(Baseline\): ([\d.]+)%', content)
        synth_acc_match = re.search(r'BEST SYNTHETIC train \u2192 REAL test accuracy: ([\d.]+)%', content)
        augmented_acc_match = re.search(r'FINAL TEST SET PERFORMANCE OF BEST STRATEGY(?: ---)?\n  Test Accuracy: ([\d.]+)%', content)

        if baseline_acc_match and synth_acc_match and augmented_acc_match:
            data.append({
                'Dataset': 'BCI2a',
                'Subject': f"S{subject}",
                'Baseline': float(baseline_acc_match.group(1)),
                'Synthetic': float(synth_acc_match.group(1)),
                'Best_Augmented': float(augmented_acc_match.group(1)),
                'Trials': trials,
                'Train Session': train_session
            })
    return data

def parse_cho2017_logs(data_dirs=["DATA1_results", "DATA2_results", "DATA3_results", "DATA4_results", "DATA5_results"]):
    """Parses logs for the Cho2017 dataset."""
    data = []
    
    def get_real_subject_num(data_dir, folder_subject_num_str):
        folder_subject_num = int(folder_subject_num_str)
        if data_dir == "DATA1_results": return folder_subject_num
        if data_dir == "DATA2_results": return folder_subject_num + 10
        if data_dir == "DATA3_results": return folder_subject_num + 20 if folder_subject_num < 9 else 30
        if data_dir == "DATA4_results": return folder_subject_num + 30 if folder_subject_num < 4 else folder_subject_num + 31
        if data_dir == "DATA5_results": return folder_subject_num + 40
        return None

    log_files = [os.path.join(root, name)
                 for data_dir in data_dirs if os.path.exists(data_dir)
                 for root, _, files in os.walk(data_dir)
                 for name in files if name == 'run_log.txt']

    for log_file in log_files:
        with open(log_file, 'r') as f:
            content = f.read()
        
        data_dir_name = os.path.basename(os.path.dirname(os.path.dirname(log_file)))
        subdir_match = re.search(r"Subject_(\d+)", os.path.dirname(log_file))
        
        if not subdir_match:
            continue
            
        folder_subject_num = subdir_match.group(1)
        subject = get_real_subject_num(data_dir_name, folder_subject_num)

        baseline_acc_match = re.search(r'BASELINE ACCURACY \(Real train\+val -> Real test\): ([\d.]+)%', content)
        synth_acc_match = re.search(r'BEST SYNTHETIC train \u2192 REAL test accuracy: ([\d.]+)%', content)
        augmented_acc_match = re.search(r'FINAL TEST SET PERFORMANCE OF BEST STRATEGY(?: ---)?\n  Test Accuracy: ([\d.]+)%', content)

        if subject and baseline_acc_match and synth_acc_match and augmented_acc_match:
            data.append({
                'Dataset': 'Cho2017',
                'Subject': f"S{subject}",
                'Baseline': float(baseline_acc_match.group(1)),
                'Synthetic': float(synth_acc_match.group(1)),
                'Best_Augmented': float(augmented_acc_match.group(1))
            })
    return data

def create_static_scatter_plots(df):
    """
    Creates and saves static scatter plots. For BCI2a, it combines trial
    counts onto a single plot, distinguished by markers.
    """
    output_dir = 'Scatter_Plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Define all plot configurations ---
    plot_configs = [
        {'name': 'P300', 'filters': (df['Dataset'] == 'P300')},
        {'name': 'Cho2017', 'filters': (df['Dataset'] == 'Cho2017')},
        # -- BCI2a Configurations: Split by training session ONLY --
        {'name': 'BCI2a_Train_S1', 'filters': (df['Dataset'] == 'BCI2a') & (df['Train Session'] == 1)},
        {'name': 'BCI2a_Train_S2', 'filters': (df['Dataset'] == 'BCI2a') & (df['Train Session'] == 2)},
    ]
    
    min_val = df[['Baseline', 'Synthetic', 'Best_Augmented']].min().min() - 5
    max_val = df[['Baseline', 'Synthetic', 'Best_Augmented']].max().max() + 5

    # Define the two types of plots to generate for each configuration
    plot_types = [
        {'x': 'Synthetic', 'title': 'Synthetic vs Baseline'},
        {'x': 'Best_Augmented', 'title': 'Augmented vs Baseline'}
    ]

    for config in plot_configs:
        # Apply the filter for the current configuration
        dataset_df = df[config['filters']]
        
        # Skip if this condition resulted in an empty dataframe
        if dataset_df.empty:
            print(f"Skipping plot for '{config['name']}' as no data was found for this condition.")
            continue

        # Generate both plot types (e.g., Synthetic vs Baseline, Augmented vs Baseline)
        for plot_type in plot_types:
            plt.figure(figsize=(10, 8))
            
            # Add the y=x identity line for easy comparison
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, label='y=x (no change)')

            # Special handling for BCI2a plots to show different trial counts
            if 'BCI2a' in config['name']:
                # Plot 25 trials data if it exists
                df_25 = dataset_df[dataset_df['Trials'] == 25]
                if not df_25.empty:
                    plt.scatter(df_25[plot_type['x']], df_25['Baseline'], alpha=0.7, marker='o', s=80, label='25 Trials')

                # Plot 50 trials data if it exists
                df_50 = dataset_df[dataset_df['Trials'] == 50]
                if not df_50.empty:
                    plt.scatter(df_50[plot_type['x']], df_50['Baseline'], alpha=0.7, marker='X', s=80, label='50 Trials')
            else:
                # Original plotting logic for P300 and Cho2017
                plt.scatter(dataset_df[plot_type['x']], dataset_df['Baseline'], alpha=0.7)

            # --- Common plot styling ---
            plot_title = f"{plot_type['title']} - {config['name'].replace('_', ' ')}"
            plt.title(plot_title)
            plt.xlabel(f"{plot_type['title'].split(' ')[0]} Accuracy (%)")
            plt.ylabel("Baseline Accuracy (%)")
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.gca().set_aspect('equal', adjustable='box') # Make the plot square
            
            # Generate a safe and descriptive filename
            filename = f"plot_{config['name']}_{plot_type['x']}.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

    print(f"\nAll scatter plots have been saved in the '{output_dir}' directory.")

def create_box_plots(df):
    """
    Creates and saves box plots comparing baseline and augmented performance for
    low and high performing groups, separated by dataset and condition.
    """
    output_dir = 'Box_Plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define plot configurations, similar to scatter plots
    plot_configs = [
        {'name': 'P300', 'filters': (df['Dataset'] == 'P300')},
        {'name': 'Cho2017', 'filters': (df['Dataset'] == 'Cho2017')},
        {'name': 'BCI2a_Train_S1', 'filters': (df['Dataset'] == 'BCI2a') & (df['Train Session'] == 1)},
        {'name': 'BCI2a_Train_S2', 'filters': (df['Dataset'] == 'BCI2a') & (df['Train Session'] == 2)},
    ]

    for config in plot_configs:
        # Apply the filter for the current dataset/condition
        dataset_df = df[config['filters']].copy()

        if dataset_df.empty:
            print(f"Skipping box plots for '{config['name']}' as no data was found for this condition.")
            continue

        # Split data into low and high performers for this specific dataset
        low_performers = dataset_df[dataset_df['Baseline'] < 70]
        high_performers = dataset_df[dataset_df['Baseline'] >= 70]

        groups = {
            'Low_Performers_Below_70_Baseline': low_performers,
            'High_Performers_Above_70_Baseline': high_performers
        }

        for group_name, group_df in groups.items():
            if group_df.empty:
                print(f"Skipping box plot for '{config['name']} - {group_name}' as no data was found.")
                continue

            # Prepare data for boxplot
            data_to_plot = [group_df['Baseline'], group_df['Best_Augmented']]
            
            plt.figure(figsize=(8, 6))
            
            bplot = plt.boxplot(data_to_plot, labels=['Baseline', 'Best Augmented'], patch_artist=True)

            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

            # Style the plot with a more descriptive title
            plot_title = f"Augmented vs Baseline for {config['name'].replace('_', ' ')}\n({group_name.replace('_', ' ')})"
            plt.title(plot_title)
            plt.ylabel('Accuracy (%)')
            plt.grid(True, linestyle='--', alpha=0.6)

            # Save the plot with a more descriptive filename
            filename = f"boxplot_{config['name']}_{group_name}.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

    print(f"\nAll box plots have been saved in the '{output_dir}' directory.")


if __name__ == '__main__':
    all_data = []
    all_data.extend(parse_p300_logs())
    all_data.extend(parse_bci2a_logs())
    all_data.extend(parse_cho2017_logs())

    if not all_data:
        print("No log files found or no data could be parsed. Exiting.")
    else:
        df = pd.DataFrame(all_data)
        
        # Drop rows if any of the crucial accuracy values are missing
        df.dropna(subset=['Baseline', 'Synthetic', 'Best_Augmented'], inplace=True)
        
        print("--- Combined and Parsed Data ---")
        print(df.to_string())
        
        if not df.empty:
            create_static_scatter_plots(df)
            create_box_plots(df)
        else:
            print("\nNo valid data remaining after cleaning. Cannot generate plots.")
