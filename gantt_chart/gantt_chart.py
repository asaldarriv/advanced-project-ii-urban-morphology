import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define activities, their periods in weeks, and completion percentage
tasks = [
    ("State of the Art Review", 1, 5, 80),
    ("Data Collection and Processing", 4, 5, 50),
    ("Calculation of Geometric and Topological Metrics", 6, 8, 0),
    ("Pattern Analysis with Machine Learning", 9, 13, 0),
    ("Development of Visualization Tools", 14, 15, 0),
    ("Validation in Colombian Cities", 16, 17, 0),
    ("Preliminary Project Development and Delivery", 1, 4, 100),
    ("Preliminary Project Presentation", 4, 4, 100),
    ("Progress Report 1", 5, 8, 0),
    ("Progress Report 2", 9, 14, 0),
    ("Final Report Advisory", 15, 16, 0),
    ("Final Report Development and Delivery", 15, 18, 0),
    ("Final Report Presentation", 18, 18, 0),
]

# Create a DataFrame with activities, their periods in weeks, and completion percentage
df = pd.DataFrame(tasks, columns=["activity", "start_week", "end_week", "completion"])

# Create a list of colors
colors = plt.cm.tab20c(np.linspace(0, 1, len(df)))

# Create the Gantt chart
fig, ax = plt.subplots(figsize=(14, 8))
for i, (task, start_week, end_week, completion) in enumerate(zip(df["activity"], df["start_week"], df["end_week"], df["completion"])):
    ax.barh(len(df) - 1 - i, end_week - start_week + 1, left=start_week - 1, height=1, align='center', color=colors[i % len(colors)])
    # Add completion percentage text
    ax.text(end_week + 0.1, len(df) - 1 - i, f'{completion}%', va='center', fontsize=10)

# Configure labels and format
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df["activity"][::-1])
ax.set_xlabel("Weeks", fontsize=14, labelpad=20)
ax.set_title("Activity Schedule - Gantt Chart", fontsize=20, pad=10)

# Move the x-axis to the top
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

# Adjust margins to avoid cutting off stage names
plt.subplots_adjust(left=0.3, top=0.85, bottom=0.15)

# Set x-axis limits to cover weeks 1 to 18
ax.set_xlim([0, 18])

# Configure x-axis to show weeks
ax.set_xticks(range(18))
ax.set_xticklabels(range(1, 19))

# Add vertical grid
ax.xaxis.grid(True, which='both')

# Save the chart as an image
plt.savefig("gantt_chart.png")

# Show the chart
plt.show()