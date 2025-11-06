import matplotlib.pyplot as plt
import numpy as np

# Data
years = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
num_papers = np.array([1, 4, 9, 14, 27, 15, 18, 12, 7])

# Create the bar chart
plt.figure(figsize=(12, 8)) # Increased figure size for better readability of larger fonts
bars = plt.bar(years, num_papers, color='skyblue')

# Add a title and labels with bigger and bolder fonts
plt.title('Trend of Papers Using GAN for EEG Data Augmentation in BCI (2017-2025)', fontsize=40, fontweight='bold')
plt.xlabel('Year', fontsize=32, fontweight='bold')
plt.ylabel('Number of Papers', fontsize=32, fontweight='bold')

# Add the value on top of each bar with bigger and bolder fonts
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', fontsize=12, fontweight='bold')

# Add a trend line to highlight the trend
plt.plot(years, num_papers, color='red', marker='o', linestyle='--')

# Customize the ticks on the x and y axes to be bigger and bolder
plt.xticks(years, fontsize=24, fontweight='bold')
plt.yticks(fontsize=24, fontweight='bold')


# Add a grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()