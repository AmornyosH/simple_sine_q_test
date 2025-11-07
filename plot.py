import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
sns.set_theme()

EXPERT_PATH = 'datasets/dataset_simple_sine.npz'
# EXPERT_PATH = 'datasets/dataset_sharp_sine.npz'
test_sample = 628

dataset = np.load(EXPERT_PATH, allow_pickle=True)  # ========== Load the dataset
x = dataset['observations']
y = dataset['actions']
r = dataset['rewards']
q = dataset['true_q']

# pred_q_gpdq_rbf = np.load('GPDQ/pred_q.npy')
# pred_q_gpdq_matern = np.load('GPDQ_matern/pred_q.npy')
# pred_q_iql = np.load('IQL/pred_q.npy')

# Load test data
# GPDQ (RBF)
gpdq_test_data = np.load('GPDQ_rbf/evaluation_data.npz')
pred_y_gpdq = gpdq_test_data['arr_0']
pred_g_mu_gpdq = gpdq_test_data['arr_1']
pred_g_var_gpdq = gpdq_test_data['arr_2']
pred_g_std_gpdq = np.sqrt(np.diagonal(pred_g_var_gpdq))
pred_q_gpdq = gpdq_test_data['arr_3']
k_nn_gpdq = gpdq_test_data['arr_4']

dql_test_data = np.load('DQL/evaluation_data.npz')
pred_y_dql = dql_test_data['arr_0']
pred_q_dql = dql_test_data['arr_3']

# IQL
iql_test_data = np.load('IQL/evaluation_data.npz')
pred_y_iql = iql_test_data['arr_0']
pred_q_iql = iql_test_data['arr_3']

# print('RMSE(GPDQ-RBF): ', np.mean(q - pred_q_gpdq))
# print('RMSE(GPDQ_32-RBF): ', np.mean(q - pred_q_gpdq32))
# print('ACCUM_REWARD(GPDQ-RBF): ', np.sum(np.abs(pred_y_gpdq[0:test_sample])))
# print('ACCUM_REWARD(IQL): ', np.sum(np.abs(pred_y_iql[0:test_sample])))
# print('ACCUM_REWARD(DQL): ', np.sum(np.abs(pred_y_dql[0:test_sample])))
# print('ACCUM_REWARD(GPDQ_32-RBF: ', np.sum(np.abs(pred_y_gpdq32[0:314])))
# print('ACCUM_REWARD(GPDQ-MATERN): ', np.sum(np.abs(pred_y_gpdq2[0:314])))
# print('ACCUM_REWARD(GPDQ-MATERNNZM): ', np.sum(np.abs(pred_y_gpdqnzm[0:314])))
print(np.min(pred_g_var_gpdq))


# Create subplots
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# Plot heatmap reward
sc = axs[0].scatter(x, y, c=r, cmap='viridis', s=50, edgecolor='k')
fig.colorbar(sc, ax=axs[0], shrink=0.8)
axs[0].set_title('Reward Function')
# axs[0].set_xlim(0, 4*np.pi)

# Plot sine wave
axs[1].scatter(x, y, alpha=0.1, color='gray', label='Ground Truth')
axs[1].scatter(x[0:test_sample], pred_y_gpdq, color='blue', label='GPDQ(RBF)')
axs[1].set_title('Non-smooth sine wave data')
axs[1].legend()
# axs[1].set_xlim(0, 4*np.pi)

# Plot GP predictions
axs[2].scatter(x, y, alpha=1, color='gray', label='Ground Truth')
axs[2].plot(x[0:test_sample], pred_g_mu_gpdq, color='blue', label='GPDQ(RBF)')
axs[2].fill_between(x[0:test_sample].flatten(), pred_g_mu_gpdq + pred_g_std_gpdq, pred_g_mu_gpdq - pred_g_std_gpdq, alpha=0.3, color='blue')
axs[2].set_title('Non-smooth sine wave data (GP predictions)')
axs[2].legend()
# axs[2].set_xlim(0, 4*np.pi)

# Plot kernel matrix
khm = axs[3].imshow(k_nn_gpdq, cmap='plasma', origin='lower')
fig.colorbar(khm, ax=axs[3], shrink=0.8)
# axs[3].set_xlim(0, len(k_nn_gpdq))
# Label x-axis only once
# fig.supxlabel('state (s)')
# fig.supylabel('action (a)')
plt.tight_layout()
plt.show()