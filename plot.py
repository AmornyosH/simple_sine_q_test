import matplotlib.pyplot as plt
import numpy as np
import torch

# EXPERT_PATH = 'datasets/dataset_simple_sine.npz'
EXPERT_PATH = 'datasets/dataset_sharp_sine.npz'

dataset = np.load(EXPERT_PATH, allow_pickle=True)  # ========== Load the dataset
x = dataset['observations']
y = dataset['actions']
r = dataset['rewards']
q = dataset['true_q']

pred_q_gpdq = np.load('GPDQ/pred_q.npy')
pred_q_iql = np.load('IQL/pred_q.npy')

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Plot sine wave
axs[0].scatter(x, y, color='blue', label='Ground Truth')
# axs[0].scatter(x[0:250], pred_y, color='purple', label='GPDQ')
axs[0].set_title('Sine Wave')
axs[0].grid(True)
axs[0].legend()

# Plot negative sine wave
axs[1].scatter(x[0:250], q, color='black', label='Ground Truth')
# axs[1].scatter(x[0:250], pred_q_gpdq, color='blue', label='GPDQ (Ours)')
# axs[1].scatter(x[0:250], pred_q_iql, color='red', label='IQL')
# axs[1].scatter(x, r, color='red', label='Rewards')
axs[1].set_title('State-Action Value Function (Q-value)')
axs[1].grid(True)
axs[1].legend()

# Label x-axis only once
plt.xlabel('x')
plt.tight_layout()
plt.show()