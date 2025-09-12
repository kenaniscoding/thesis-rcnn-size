import matplotlib.pyplot as plt
import numpy as np

# Data
epochs = np.arange(1, 26)

# Training loss data
train_loss = [0.7011, 0.3358, 0.2735, 0.2185, 0.1751, 0.1354, 0.1199, 0.0978, 
              0.0828, 0.0949, 0.0664, 0.0578, 0.0518, 0.0593, 0.0390, 0.0314, 
              0.0260, 0.0426, 0.0303, 0.0253, 0.0277, 0.0258, 0.0322, 0.0189, 0.0143]

# Validation loss data
val_loss = [0.3767, 0.3065, 0.2783, 0.2660, 0.2192, 0.2448, 0.2043, 0.2020, 
            0.3241, 0.2059, 0.3161, 0.3198, 0.2494, 0.2068, 0.1905, 0.2745, 
            0.2757, 0.2756, 0.3118, 0.3164, 0.2675, 0.3195, 0.3164, 0.3281, 0.3142]

# Train accuracy data
train_accuracy = [0.75, 0.88, 0.90, 0.92, 0.95, 0.96, 0.97, 0.98, 0.985, 0.983,
                  0.983, 0.982, 0.983, 0.981, 0.985, 0.987, 0.988, 0.986, 0.988,
                  0.99, 0.993, 0.997, 0.995, 0.996, 0.997]

# train_accuracy = [0.75, 0.88, 0.90, 0.92, 0.95, 0.96, 0.97, 0.98, 0.985, 0.983, 0.98, 
#                   0.982, 0.983, 0.981, 0.985, 0.987, 0.988, 0.986, 
#                   0.988, 0.989, 0.99, 0.992, 0.996, 0.995, 0.997]
# 0.991, 0.995, 0.997, 0.998, 0.996, 0.998, 0.999, 0.998, 0.999, 0.997, 0.9995, 1.0]
# Validation accuracy data
val_accuracy = [0.8770, 0.8843, 0.8897, 0.9295, 0.9458, 0.9458, 0.9548, 0.9656, 
                0.9259, 0.9566, 0.9512, 0.9548, 0.9548, 0.9693, 0.9675, 0.9656, 
                0.9638, 0.9494, 0.9548, 0.9638, 0.9638, 0.9620, 0.9530, 0.9620, 0.9638]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Loss comparison
ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=4)
ax1.plot(epochs, val_loss, 'r-o', label='Validation Loss', linewidth=2, markersize=4)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training vs Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, 25)

# Plot 2: Accuracy comparison
ax2.plot(epochs, train_accuracy, 'g-o', label='Training Accuracy', linewidth=2, markersize=4)
ax2.plot(epochs, val_accuracy, 'orange', marker='o', label='Validation Accuracy', linewidth=2, markersize=4)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training vs Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 25)
ax2.set_ylim(0.7, 1.0)

# Adjust layout and show
plt.tight_layout()
plt.show()

# Print summary statistics
print("=== Training Results Summary ===")
print(f"Final Training Loss: {train_loss[-1]:.4f}")
print(f"Final Validation Loss: {val_loss[-1]:.4f}")
print(f"Final Training Accuracy: {train_accuracy[-1]:.4f}")
print(f"Final Validation Accuracy: {val_accuracy[-1]:.4f}")

print(f"\nBest Validation Loss: {min(val_loss):.4f} at epoch {val_loss.index(min(val_loss)) + 1}")
print(f"Best Validation Accuracy: {max(val_accuracy):.4f} at epoch {val_accuracy.index(max(val_accuracy)) + 1}")

# Calculate overfitting indicators
loss_gap = val_loss[-1] - train_loss[-1]
acc_gap = train_accuracy[-1] - val_accuracy[-1]
print(f"\nOverfitting Indicators:")
print(f"Final Loss Gap (Val - Train): {loss_gap:.4f}")
print(f"Final Accuracy Gap (Train - Val): {acc_gap:.4f}")
