import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from scipy.stats import ttest_rel

# --- 1. FINAL COMBINED DATA STRUCTURE ---
# All data is now organized by top-level dataset, then by experimental condition.

all_data = {
    'Cho2017': {
        '40 Trials/Class': {
            'S1': [80.00, 73.33, 81.67, 80.00, 82.50], 'S2': [52.50, 60.83, 49.17, 51.67, 50.83],
            'S3': [98.33, 85.83, 97.50, 95.00, 95.00], 'S4': [81.67, 73.33, 84.17, 80.00, 75.83],
            'S5': [63.87, 67.23, 64.71, 58.82, 57.14], 'S6': [69.39, 65.31, 66.33, 63.27, 65.31],
            'S7': [55.62, 56.88, 56.25, 53.12, 54.37], 'S8': [52.54, 62.71, 58.47, 55.08, 50.85],
            'S9': [69.38, 63.12, 65.62, 61.25, 58.13], 'S10': [80.00, 66.67, 82.50, 85.00, 81.67],
            'S11': [63.33, 60.00, 63.33, 65.00, 66.67], 'S12': [73.40, 67.02, 70.21, 67.02, 69.15],
            'S13': [65.83, 65.83, 63.33, 65.83, 65.83], 'S14': [98.33, 90.83, 98.33, 98.33, 95.00],
            'S15': [66.67, 60.36, 65.77, 65.77, 68.47], 'S16': [50.42, 61.34, 52.10, 60.50, 58.82],
            'S17': [55.00, 62.50, 52.50, 57.50, 59.17], 'S18': [62.50, 61.67, 60.83, 69.17, 71.67],
            'S19': [57.14, 71.43, 63.49, 61.90, 68.25], 'S21': [75.63, 68.91, 70.59, 67.23, 63.03],
            'S22': [75.00, 64.17, 75.00, 74.17, 73.33], 'S23': [87.18, 67.52, 88.03, 82.05, 82.05],
            'S24': [70.59, 61.34, 66.39, 68.07, 67.23], 'S25': [62.93, 62.93, 61.21, 58.62, 61.21],
            'S26': [60.83, 57.50, 60.83, 60.83, 60.83], 'S27': [61.90, 63.81, 65.71, 64.76, 61.90],
            'S28': [56.03, 59.48, 57.76, 61.21, 62.93], 'S30': [73.81, 69.05, 76.19, 76.19, 71.43],
            'S31': [69.75, 61.34, 68.07, 64.71, 62.18], 'S32': [55.00, 60.83, 51.67, 50.83, 55.00],
            'S33': [58.82, 60.50, 53.78, 53.78, 52.10], 'S35': [80.00, 71.67, 81.67, 72.50, 73.33],
            'S36': [78.26, 65.22, 81.74, 82.61, 80.87], 'S37': [65.83, 65.00, 68.33, 68.33, 70.00],
            'S38': [47.75, 60.36, 50.45, 48.65, 48.65], 'S39': [64.71, 58.82, 59.66, 57.14, 57.98],
            'S40': [55.83, 61.67, 54.17, 50.83, 54.17], 'S41': [89.08, 75.63, 85.71, 86.55, 87.39],
            'S42': [64.17, 61.67, 57.50, 63.33, 60.83], 'S43': [97.50, 80.83, 95.00, 95.83, 96.67],
            'S44': [64.17, 63.33, 62.50, 59.17, 58.33], 'S45': [54.17, 61.67, 51.67, 47.50, 48.33],
            'S46': [78.62, 62.89, 78.62, 76.73, 76.73], 'S47': [66.96, 67.86, 68.75, 67.86, 64.29],
            'S48': [84.87, 68.07, 77.31, 79.83, 78.99], 'S49': [71.43, 62.50, 68.75, 71.43, 70.54],
            'S50': [58.33, 61.67, 55.83, 51.67, 57.50], 'S51': [64.91, 61.40, 65.79, 64.04, 65.79],
            'S52': [70.83, 59.17, 73.33, 69.17, 65.00],
        }
    },
    'BCI_2a': {
        '20 Trials/Class': {
            'S1': [86.81, 77.78, 89.58, 88.89, 77.78], 'S2': [60.42, 59.03, 56.94, 53.47, 56.94],
            'S3': [93.75, 86.11, 87.50, 85.42, 86.81], 'S4': [64.58, 60.42, 59.72, 65.97, 65.28],
            'S5': [62.50, 61.81, 60.42, 61.11, 59.72], 'S6': [62.50, 59.03, 62.50, 63.89, 63.89],
            'S7': [72.22, 61.81, 70.14, 57.64, 60.42], 'S8': [93.75, 78.47, 88.89, 89.58, 88.89],
            'S9': [91.67, 89.58, 91.67, 92.36, 91.67],
        },
        '50 Trials/Class': {
            'S1': [87.50, 72.92, 88.19, 88.89, 86.81], 'S2': [54.86, 59.03, 50.00, 51.39, 53.47],
            'S3': [97.22, 81.25, 96.53, 96.53, 97.22], 'S4': [68.06, 63.19, 67.36, 66.67, 68.06],
            'S5': [68.75, 60.42, 66.67, 68.06, 55.56], 'S6': [67.36, 59.03, 63.89, 63.19, 64.58],
            'S7': [70.83, 65.28, 67.36, 69.44, 72.92], 'S8': [95.14, 85.42, 94.44, 93.06, 93.06],
            'S9': [92.36, 88.19, 92.36, 91.67, 93.06],
        },
        '20 trials/class': {
            'S1': [76.81, 65.94, 84.78, 76.81, 76.09], 'S2': [54.41, 63.24, 52.94, 58.09, 50.74],
            'S3': [94.16, 75.18, 90.51, 94.89, 90.51], 'S4': [62.79, 61.24, 65.89, 65.89, 68.22],
            'S5': [67.44, 61.24, 61.24, 55.04, 55.81], 'S6': [69.91, 61.95, 57.52, 61.95, 63.72],
            'S7': [63.16, 61.65, 61.65, 60.90, 60.15], 'S8': [93.94, 87.12, 91.67, 90.15, 84.85],
            'S9': [87.07, 81.03, 87.07, 84.48, 91.38],
        },
        '50 trials/class': {
            'S1': [79.71, 68.84, 81.16, 79.71, 80.43], 'S2': [52.21, 58.82, 45.59, 52.94, 52.21],
            'S3': [93.43, 81.75, 92.70, 91.97, 91.24], 'S4': [66.67, 62.79, 68.22, 67.44, 65.12],
            'S5': [77.52, 60.47, 70.54, 71.32, 68.22], 'S6': [67.26, 69.03, 73.45, 70.80, 72.57],
            'S7': [68.42, 60.90, 63.16, 66.17, 66.17], 'S8': [95.45, 79.55, 95.45, 93.94, 93.94],
            'S9': [89.66, 73.28, 84.48, 88.79, 81.03],
        }
    },
    'Jin_P300': {
        '30 Trials/Class': {
            'S1': [70.00, 82.86, 70.00, 72.86, 70.00],
            'S2': [71.43, 85.71, 67.14, 62.86, 82.86],
        }
    }
}

# --- 2. PREPARE ALL POSSIBLE VIEWS FOR THE DROPDOWN ---
categories = ['Real', 'Synthetic', 'Real + 25% Synth', 'Real + 50% Synth', 'Real + 100% Synth']
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
plots_per_page = 10
grid_rows, grid_cols = 5, 2

all_views = []

for dataset_name, conditions in all_data.items():
    for condition_name, subject_data in conditions.items():
        all_subject_keys = sorted(subject_data.keys(), key=lambda s: int(s[1:]))
        num_pages = math.ceil(len(all_subject_keys) / plots_per_page)

        for page_num in range(num_pages):
            start_idx = page_num * plots_per_page
            end_idx = start_idx + plots_per_page
            page_subjects = all_subject_keys[start_idx:end_idx]

            view = {"subjects": page_subjects, "y_data": [], "text_data": [], "subplot_titles": []}

            # --- Generate the custom title based on your rules ---
            if dataset_name == 'Cho2017':
                view['title'] = '40 Trials/Class Training & Rest Testing'
            elif dataset_name == 'Jin_P300':
                view['title'] = '30 Trials/Class Training & Rest Testing'
            elif dataset_name == 'BCI_2a':
                if condition_name in ['20 Trials/Class', '50 Trials/Class']:
                    view['title'] = f'{condition_name} - Session_1 (Training) & Session_2 (Testing)'
                else: # 100 and 200 Trials/Class
                    view['title'] = f'{condition_name} - Session_1 (Testing) & Session_2 (Training)'

            view['dataset'] = dataset_name
            view['condition'] = condition_name

            for i in range(plots_per_page):
                if i < len(page_subjects):
                    subject_key = page_subjects[i]
                    accuracies = subject_data[subject_key]
                    view["y_data"].append(accuracies)
                    view["text_data"].append([f'{acc:.2f}' for acc in accuracies])
                    view["subplot_titles"].append(f'{subject_key}: Accuracy')
                else:
                    view["y_data"].append([]); view["text_data"].append([]); view["subplot_titles"].append("")
            all_views.append(view)

# --- 3. CREATE THE FIGURE WITH SUBPLOTS ---
initial_view = all_views[0]
fig = make_subplots(rows=grid_rows, cols=grid_cols, subplot_titles=initial_view["subplot_titles"])

for i in range(plots_per_page):
    row, col = i // grid_cols + 1, i % grid_cols + 1
    fig.add_trace(go.Bar(x=categories, marker_color=colors, textposition='auto'), row=row, col=col)

# --- 4. CREATE THE UNIFIED DROPDOWN MENU ---
dropdown_buttons = []
for view in all_views:
    page_label = f"Subject {view['subjects'][0][1:]}-{view['subjects'][-1][1:]}"
    dataset_name = view['dataset']
    condition_name = view['condition']

    # --- Generate the custom dropdown label based on your rules ---
    if dataset_name in ['Cho2017', 'Jin_P300']:
        label = f'{dataset_name} - {page_label}'
    else: # BCI_2a
        condition_short = condition_name.split(" ")[0]
        if condition_name in ['20 Trials/Class', '50 Trials/Class']:
            train_session_label = "Train Se1"
        else: # 100 and 200
            train_session_label = "Train Se2"
        label = f'BCI_2a ({condition_short} Trials, {train_session_label}) - {page_label}'

    layout_updates = {f'annotations[{j}].text': title for j, title in enumerate(view["subplot_titles"])}
    layout_updates['title.text'] = view['title']

    button = dict(label=label, method='update', args=[{'y': view["y_data"], 'text': view["text_data"]}, layout_updates])
    dropdown_buttons.append(button)

fig.update_layout(updatemenus=[dict(active=0, buttons=dropdown_buttons, direction="down",
                                    pad={"r": 10, "t": 10}, showactive=True,
                                    x=0.01, xanchor="left", y=1.07, yanchor="top")])

# --- 5. FINALIZE LAYOUT AND STYLING ---
fig.update_traces(y=initial_view['y_data'], text=initial_view['text_data'])
fig.update_layout(
    title_text=initial_view['title'],
    title_x=0.5, height=1400, showlegend=False, font=dict(size=12)
)

for i in range(1, plots_per_page + 1):
     fig.update_yaxes(range=[0, 100], row=(i-1)//grid_cols + 1, col=(i-1)%grid_cols + 1)

# --- 6. SAVE AND SHOW THE PLOT ---
output_filename = 'complete_interactive_report.html'
fig.write_html(output_filename)
fig.show()

print(f"\nSuccessfully generated the final interactive plot with all datasets and custom rules.")
print(f"The file is saved as '{output_filename}' and should have opened in your browser.")

# --- 7. CREATE AND SHOW COMBINED SCATTER PLOTS ---
scatter_fig = go.Figure()

# Define more granular BCI_2a subgroups
bci2a_subgroups = {
    "BCI_2a (20 Trials, Train Se1)": ['20 Trials/Class'],
    "BCI_2a (50 Trials, Train Se1)": ['50 Trials/Class'],
    "BCI_2a (20 Trials, Train Se2)": ['20 trials/class'],
    "BCI_2a (50 Trials, Train Se2)": ['50 trials/class']
}
datasets_for_scatter = {'Cho2017': all_data.get('Cho2017', {}), 'Jin_P300': all_data.get('Jin_P300', {})}

# --- Add traces for "Real vs. Synthetic" plot (Visible by default) ---
all_real_scores_1, all_synthetic_scores_1 = [], []
for dataset_name, conditions in datasets_for_scatter.items():
    if conditions:
        real, synthetic, texts = [], [], []
        for cond, subjects in conditions.items():
            for subj, accs in subjects.items():
                real.append(accs[0]); synthetic.append(accs[1])
                all_real_scores_1.append(accs[0]); all_synthetic_scores_1.append(accs[1])
                texts.append(f"S: {subj}<br>C: {cond}<br>R: {accs[0]:.2f}%<br>S: {accs[1]:.2f}%")
        scatter_fig.add_trace(go.Scatter(
            x=synthetic, y=real, mode='markers', name=dataset_name, text=texts,
            hoverinfo='text', marker=dict(size=8, opacity=0.8), visible=True
        ))

if 'BCI_2a' in all_data:
    for group, cond_list in bci2a_subgroups.items():
        real, synthetic, texts = [], [], []
        for cond in cond_list:
            if cond in all_data['BCI_2a']:
                for subj, accs in all_data['BCI_2a'][cond].items():
                    real.append(accs[0]); synthetic.append(accs[1])
                    all_real_scores_1.append(accs[0]); all_synthetic_scores_1.append(accs[1])
                    texts.append(f"S: {subj}<br>C: {cond}<br>R: {accs[0]:.2f}%<br>S: {accs[1]:.2f}%")
        if real:
            scatter_fig.add_trace(go.Scatter(
                x=synthetic, y=real, mode='markers', name=group, text=texts,
                hoverinfo='text', marker=dict(size=8, opacity=0.8), visible=True
            ))

# --- Add traces for "Real vs. Best Augmented" plot (Invisible by default) ---
all_real_scores_2, all_best_aug_scores_2 = [], []
for dataset_name, conditions in datasets_for_scatter.items():
    if conditions:
        real, best_aug, texts = [], [], []
        for cond, subjects in conditions.items():
            for subj, accs in subjects.items():
                real.append(accs[0]); best_aug.append(max(accs[1:]))
                all_real_scores_2.append(accs[0]); all_best_aug_scores_2.append(max(accs[1:]))
                texts.append(f"S: {subj}<br>C: {cond}<br>R: {accs[0]:.2f}%<br>A: {max(accs[1:]):.2f}%")
        scatter_fig.add_trace(go.Scatter(
            x=best_aug, y=real, mode='markers', name=dataset_name, text=texts,
            hoverinfo='text', marker=dict(size=8, opacity=0.8), visible=False
        ))

if 'BCI_2a' in all_data:
    for group, cond_list in bci2a_subgroups.items():
        real, best_aug, texts = [], [], []
        for cond in cond_list:
            if cond in all_data['BCI_2a']:
                for subj, accs in all_data['BCI_2a'][cond].items():
                    real.append(accs[0]); best_aug.append(max(accs[1:]))
                    all_real_scores_2.append(accs[0]); all_best_aug_scores_2.append(max(accs[1:]))
                    texts.append(f"S: {subj}<br>C: {cond}<br>R: {accs[0]:.2f}%<br>A: {max(accs[1:]):.2f}%")
        if real:
            scatter_fig.add_trace(go.Scatter(
                x=best_aug, y=real, mode='markers', name=group, text=texts,
                hoverinfo='text', marker=dict(size=8, opacity=0.8), visible=False
            ))

# --- Create dropdown menu to switch between scatter plots ---
num_base_traces = len(datasets_for_scatter) + len(bci2a_subgroups)
visibility_synthetic = [True] * num_base_traces + [False] * num_base_traces
visibility_augmented = [False] * num_base_traces + [True] * num_base_traces

min_val_1 = min(all_real_scores_1 + all_synthetic_scores_1) - 5 if all_real_scores_1 else 40
min_val_2 = min(all_real_scores_2 + all_best_aug_scores_2) - 5 if all_real_scores_2 else 40

scatter_fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Real vs. Synthetic",
                     method="update",
                     args=[{"visible": visibility_synthetic},
                           {"title": "Real vs. Synthetic Data Classification Accuracy",
                            "xaxis.title": "Synthetic Data Only Accuracy (%)",
                            "xaxis.range": [min_val_1, 102],
                            "yaxis.range": [min_val_1, 102]}]),
                dict(label="Real vs. Best Augmented",
                     method="update",
                     args=[{"visible": visibility_augmented},
                           {"title": "Real vs. Best Augmented Data Classification Accuracy",
                            "xaxis.title": "Best Augmented Data Accuracy (%)",
                            "xaxis.range": [min_val_2, 102],
                            "yaxis.range": [min_val_2, 102]}])
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.01,
            xanchor="left",
            y=1.1,
            yanchor="top"
        )
    ]
)

# --- Finalize layout for the initial view ---
scatter_fig.update_layout(
    title='Real vs. Synthetic Data Classification Accuracy',
    xaxis_title='Synthetic Data Only Accuracy (%)',
    yaxis_title='Real Data Only Accuracy (%)',
    legend_title='Dataset',
    xaxis=dict(range=[min_val_1, 102]),
    yaxis=dict(range=[min_val_1, 102]),
    width=900,
    height=800,
    legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1)
)

scatter_fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="grey", width=1, dash="dash"))

# --- Save and show the combined scatter plot ---
scatter_output_filename = 'combined_scatter_plots.html'
scatter_fig.write_html(scatter_output_filename)
scatter_fig.show()

print(f"\nSuccessfully generated the combined scatter plot.")
print(f"The file is saved as '{scatter_output_filename}' and should have opened in your browser.")

# --- 9. STATISTICAL ANALYSIS ---
print("\n\n--- STATISTICAL ANALYSIS ---\n")

# Define the same subgroups for consistency
bci2a_subgroups_stat = {
    "BCI_2a (20 Trials, Train Se1)": ['20 Trials/Class'],
    "BCI_2a (50 Trials, Train Se1)": ['50 Trials/Class'],
    "BCI_2a (20 Trials, Train Se2)": ['20 trials/class'],
    "BCI_2a (50 Trials, Train Se2)": ['50 trials/class']
}
datasets_for_stat = {'Cho2017': all_data.get('Cho2017', {}), 'Jin_P300': all_data.get('Jin_P300', {})}

def run_and_print_t_test(group_name, real_scores, synthetic_scores, augmented_100_scores, best_augmented_scores):
    """Helper function to run t-tests and print results."""
    print(f"--- {group_name} ---")
    if len(real_scores) < 2:
        print("Not enough data for statistical analysis.\n")
        return

    # 1. Real vs. Synthetic
    t_stat_synth, p_val_synth = ttest_rel(real_scores, synthetic_scores)
    print(f"Real vs. Synthetic: t-statistic = {t_stat_synth:.4f}, p-value = {p_val_synth:.4f}")

    # 2. Real vs. 100% Augmented
    t_stat_aug100, p_val_aug100 = ttest_rel(real_scores, augmented_100_scores)
    print(f"Real vs. 100% Augmented: t-statistic = {t_stat_aug100:.4f}, p-value = {p_val_aug100:.4f}")

    # 3. Real vs. Best Augmented
    t_stat_best_aug, p_val_best_aug = ttest_rel(real_scores, best_augmented_scores)
    print(f"Real vs. Best Augmented: t-statistic = {t_stat_best_aug:.4f}, p-value = {p_val_best_aug:.4f}\n")


# --- Process Cho2017 and Jin_P300 ---
for dataset_name, conditions in datasets_for_stat.items():
    if conditions:
        real, synthetic, aug100, best_aug = [], [], [], []
        for cond, subjects in conditions.items():
            for subj, accs in subjects.items():
                real.append(accs[0])
                synthetic.append(accs[1])
                aug100.append(accs[4]) # 100% augmented is at index 4
                best_aug.append(max(accs[1:]))
        run_and_print_t_test(dataset_name, real, synthetic, aug100, best_aug)


# --- Process BCI_2a subgroups ---
if 'BCI_2a' in all_data:
    for group, cond_list in bci2a_subgroups_stat.items():
        real, synthetic, aug100, best_aug = [], [], [], []
        for cond in cond_list:
            if cond in all_data['BCI_2a']:
                for subj, accs in all_data['BCI_2a'][cond].items():
                    real.append(accs[0])
                    synthetic.append(accs[1])
                    aug100.append(accs[4])
                    best_aug.append(max(accs[1:]))
        if real:
            run_and_print_t_test(group, real, synthetic, aug100, best_aug)

