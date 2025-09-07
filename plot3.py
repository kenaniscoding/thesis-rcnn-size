import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "V2 L diff": [1.06, 1.89, 1.02, 1.58, 2.23, 2.62, 2.36, 2.26, 2.2, 1.4, 2.6, 1.5, 2.08, 2.21],
    "V2 W diff": [0.16, 0.5, 0, 1.17, 0.69, 0.47, 0.32, 0.51, 0.45, 0.45, 1.45, 0.48, 0.27, 0.25],
    "V1 L diff": [1.47, 1.21, 2.23, 1.31, 1.38, 1.32, 0.8, 1.28, 0.88, 1.63, 1.7, 1.92, 2, 1.62],
    "V1 W diff": [4.72, 4.07, 5.24, 5.5, 4.54, 4.62, 4.2, 3.74, 3.74, 4.91, 5.74, 5.02, 4.42, 3.39],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create sample indices
x = np.arange(len(df))
width = 0.2  # Width of bars

# Plot Length Differences
plt.figure(figsize=(14, 8))
bars1 = plt.bar(x - width/2, df["V2 L diff"], width, label="V2 Length Diff", alpha=0.8, color='orange')
bars2 = plt.bar(x + width/2, df["V1 L diff"], width, label="V1 Length Diff", alpha=0.8, color='green')

# Add value labels on bars
for i, bar in enumerate(bars1):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05, f'{height:.2f}', 
             ha='center', va='bottom', fontsize=8)

for i, bar in enumerate(bars2):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05, f'{height:.2f}', 
             ha='center', va='bottom', fontsize=8)

plt.title("Length Absolute Difference from Ground Truth")
plt.xlabel("Sample Index")
plt.ylabel("Absolute Difference (cm)")
plt.xticks(x, range(len(df)))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot Width Differences
plt.figure(figsize=(14, 8))
bars1 = plt.bar(x - width/2, df["V2 W diff"], width, label="V2 Width Diff", alpha=0.8, color='orange')
bars2 = plt.bar(x + width/2, df["V1 W diff"], width, label="V1 Width Diff", alpha=0.8, color='green')

# Add value labels on bars
for i, bar in enumerate(bars1):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.2f}', 
             ha='center', va='bottom', fontsize=8)

for i, bar in enumerate(bars2):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.2f}', 
             ha='center', va='bottom', fontsize=8)

plt.title("Width Absolute Difference from Ground Truth")
plt.xlabel("Sample Index")
plt.ylabel("Absolute Difference (cm)")
plt.xticks(x, range(len(df)))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary statistics
print("Summary Statistics:")
print("Length Differences:")
print(f"V2 Code - Mean: {df['V2 L diff'].mean():.2f} cm, Std: {df['V2 L diff'].std():.2f} cm")
print(f"V1 Code - Mean: {df['V1 L diff'].mean():.2f} cm, Std: {df['V1 L diff'].std():.2f} cm")

print("\nWidth Differences:")
print(f"V2 Code - Mean: {df['V2 W diff'].mean():.2f} cm, Std: {df['V2 W diff'].std():.2f} cm")
print(f"V1 Code - Mean: {df['V1 W diff'].mean():.2f} cm, Std: {df['V1 W diff'].std():.2f} cm")