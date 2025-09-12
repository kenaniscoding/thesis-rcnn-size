import matplotlib.pyplot as plt
import numpy as np

# Extract data from the bruises classification log
epochs = np.arange(1, 8)  # 7 epochs total (early stopping at epoch 7)

# Training loss data (Average Loss)
train_loss = [0.1731, 0.0724, 0.0293, 0.0113, 0.0063, 0.0709, 0.0332]

# Validation loss data
val_loss = [0.0756, 0.0300, 0.0596, 0.0344, 0.0372, 0.0669, 0.0574]

# Validation accuracy data
val_accuracy = [0.9765, 0.9922, 0.9791, 0.9869, 0.9896, 0.9713, 0.9869]

# Training accuracy data
train_accuracy = [0.92, 0.975, 0.985, 0.995, 0.998, 0.957, 0.97]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Loss comparison
ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=6)
ax1.plot(epochs, val_loss, 'r-o', label='Validation Loss', linewidth=2, markersize=6)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Bruises Classification: Training vs Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, 7)
ax1.set_xticks(epochs)

# Plot 2: Accuracy comparison
ax2.plot(epochs, train_accuracy, 'g-o', label='Training Accuracy', linewidth=2, markersize=6)
ax2.plot(epochs, val_accuracy, 'orange', marker='o', label='Validation Accuracy', linewidth=2, markersize=6)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Bruises Classification: Training vs Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 7)
ax2.set_xticks(epochs)
ax2.set_ylim(0.9, 1.0)

# Adjust layout and show
plt.tight_layout()
plt.show()

# Print summary statistics
print("=== Bruises Classification Training Results Summary ===")
print(f"Final Training Loss: {train_loss[-1]:.4f}")
print(f"Final Validation Loss: {val_loss[-1]:.4f}")
print(f"Final Training Accuracy: {train_accuracy[-1]:.4f}")
print(f"Final Validation Accuracy: {val_accuracy[-1]:.4f}")

print(f"\nBest Training Loss: {min(train_loss):.4f} at epoch {train_loss.index(min(train_loss)) + 1}")
print(f"Best Validation Loss: {min(val_loss):.4f} at epoch {val_loss.index(min(val_loss)) + 1}")
print(f"Best Training Accuracy: {max(train_accuracy):.4f} at epoch {train_accuracy.index(max(train_accuracy)) + 1}")
print(f"Best Validation Accuracy: {max(val_accuracy):.4f} at epoch {val_accuracy.index(max(val_accuracy)) + 1}")

# Calculate performance indicators
loss_gap = val_loss[-1] - train_loss[-1]
acc_gap = train_accuracy[-1] - val_accuracy[-1]
print(f"\nPerformance Analysis:")
print(f"Final Loss Gap (Val - Train): {loss_gap:.4f}")
print(f"Final Accuracy Gap (Train - Val): {acc_gap:.4f}")

# Identify potential issues
if train_loss[4] < train_loss[5]:  # Check if training loss increased at epoch 6
    print(f"\nNote: Training loss increased significantly from epoch 5 to 6:")
    print(f"  Epoch 5: {train_loss[4]:.4f} → Epoch 6: {train_loss[5]:.4f}")
    print("  This suggests possible training instability or learning rate issues.")

if val_accuracy[1] > val_accuracy[-1]:  # Check if validation accuracy peaked early
    print(f"\nNote: Best validation accuracy was at epoch 2 ({val_accuracy[1]:.4f})")
    print("  Early stopping could have been applied around epoch 2-3.")
