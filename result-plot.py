import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Put your data into a dictionary
data = {
    "Weight (g)": [260.8, 299.4, 238.4, 335.6, 272.4, 267.9, 274, 272.3, 281.6, 286.2, 284.6, 265.7, 278.1, 263.8],
    "GT Length": [11.8, 12.8, 11.4, 13.8, 12.9, 13.1, 12.6, 13.3, 13.0, 13.8, 12.6, 13.3, 13.0, 12.9],
    "GT Width": [7.8, 7.8, 7.6, 10.5, 8.5, 8.2, 8.2, 8.0, 8.0, 8.0, 9.0, 8.0, 7.6, 7.5],
    "V2 Length": [12.86, 14.69, 12.42, 15.38, 15.13, 15.72, 14.96, 15.56, 15.2, 15.2, 15.2, 14.8, 15.08, 15.11],
    "V2 Width":  [7.96, 8.3, 7.6, 9.33, 7.81, 7.73, 7.88, 7.49, 7.55, 7.55, 7.55, 8.48, 7.87, 7.75],
    "V1 Length": [10.33, 11.59, 9.17, 12.49, 11.52, 11.78, 11.8, 12.02, 12.12, 12.17, 10.9, 11.38, 11.0, 11.28],
    "V1 Width":  [3.08, 3.73, 2.36, 5.0, 3.96, 3.58, 4.0, 4.26, 4.26, 3.09, 3.26, 2.98, 3.18, 4.11],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate percent differences
df["V1 Length % Diff"] = ((df["V1 Length"] - df["GT Length"]) / df["GT Length"]) * 100
df["V2 Length % Diff"] = ((df["V2 Length"] - df["GT Length"]) / df["GT Length"]) * 100
df["V1 Width % Diff"] = ((df["V1 Width"] - df["GT Width"]) / df["GT Width"]) * 100
df["V2 Width % Diff"] = ((df["V2 Width"] - df["GT Width"]) / df["GT Width"]) * 100

# Create sample indices
x = np.arange(len(df))
width = 0.25  # Width of bars

# Plot Length comparison as bar graph
plt.figure(figsize=(12, 6))
plt.bar(x - width, abs(df["GT Length"]), width, label="Ground Truth", alpha=0.8)
plt.bar(x, abs(df["V2 Length"]), width, label="V2 Code", alpha=0.8)
plt.bar(x + width, abs(df["V1 Length"]), width, label="V1 Code", alpha=0.8)
plt.title("Length Comparison")
plt.xlabel("Mango Index")
plt.ylabel("Length (cm)")
plt.xticks(x, range(len(df)))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot Width comparison as bar graph
plt.figure(figsize=(12, 6))
plt.bar(x - width, abs(df["GT Width"]), width, label="Ground Truth", alpha=0.8)
plt.bar(x, abs(df["V2 Width"]), width, label="V2 Code", alpha=0.8)
plt.bar(x + width, abs(df["V1 Width"]), width, label="V1 Code", alpha=0.8)
plt.title("Width Comparison")
plt.xlabel("Mango Index")
plt.ylabel("Width (cm)")
plt.xticks(x, range(len(df)))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot Length Percent Difference
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, abs(df["V2 Length % Diff"]), width, label="V2 Code", alpha=0.8, color='orange')
plt.bar(x + width/2, abs(df["V1 Length % Diff"]), width, label="V1 Code", alpha=0.8, color='green')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title("Length Percent Difference from Ground Truth")
plt.xlabel("Mango Index")
plt.ylabel("Percent Difference (%)")
plt.xticks(x, range(len(df)))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot Width Percent Difference
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, abs(df["V2 Width % Diff"]), width, label="V2 Code", alpha=0.8, color='orange')
plt.bar(x + width/2, abs(df["V1 Width % Diff"]), width, label="V1 Code", alpha=0.8, color='green')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title("Width Percent Difference from Ground Truth")
plt.xlabel("Mango Index")
plt.ylabel("Percent Difference (%)")
plt.xticks(x, range(len(df)))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary statistics
print("Length Absolute Percent Difference Summary:")
print(f"V1 Code - Mean: {df['V1 Length % Diff'].mean():.2f}%, Std: {df['V1 Length % Diff'].std():.2f}%")
print(f"V2 Code - Mean: {df['V2 Length % Diff'].mean():.2f}%, Std: {df['V2 Length % Diff'].std():.2f}%")

print("\nWidth Absolute Percent Difference Summary:")
print(f"V1 Code - Mean: {df['V1 Width % Diff'].mean():.2f}%, Std: {df['V1 Width % Diff'].std():.2f}%")
print(f"V2 Code - Mean: {df['V2 Width % Diff'].mean():.2f}%, Std: {df['V2 Width % Diff'].std():.2f}%")