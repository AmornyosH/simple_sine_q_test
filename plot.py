import matplotlib.pyplot as plt
import numpy as np
import torch
# import seaborn as sns
# sns.set_theme()

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
# GPDQ (RBF) -- Full data
gpdq_test_data = np.load('GPDQ_rbf/evaluation_data.npz')
pred_y_gpdq = gpdq_test_data['arr_0']
pred_g_mu_gpdq = gpdq_test_data['arr_1']
pred_g_var_gpdq = gpdq_test_data['arr_2']
pred_g_std_gpdq = np.sqrt(np.diagonal(pred_g_var_gpdq))
pred_q_gpdq = gpdq_test_data['arr_3']
k_nn_gpdq = gpdq_test_data['arr_4']
pred_r_gpdq = np.zeros([len(pred_y_gpdq)])
for j in range(len(pred_y_gpdq)):
    if x[j] <= np.pi or x[j] >= 3 * np.pi:
        pred_r_gpdq[j] = pred_y_gpdq[j]
    else:
        pred_r_gpdq[j] = -pred_y_gpdq[j]


# GPDQ (RBF) -- Half data
gpdq2_test_data = np.load('GPDQ_rbf/evaluation_data_2.npz')
pred_y_gpdq2 = gpdq2_test_data['arr_0']
pred_g_mu_gpdq2 = gpdq2_test_data['arr_1']
pred_g_var_gpdq2 = gpdq2_test_data['arr_2']
pred_g_std_gpdq2 = np.sqrt(np.diagonal(pred_g_var_gpdq2))
pred_q_gpdq2 = gpdq2_test_data['arr_3']
k_nn_gpdq2 = gpdq2_test_data['arr_4']
pred_r_gpdq2 = np.zeros([len(pred_y_gpdq2)])
for j in range(len(pred_y_gpdq2)):
    if x[j] <= np.pi or x[j] >= 3 * np.pi:
        pred_r_gpdq2[j] = pred_y_gpdq2[j]
    else:
        pred_r_gpdq2[j] = -pred_y_gpdq2[j]

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
fig, axs = plt.subplots(2, 4, figsize=(19, 6), constrained_layout=True)
# Plot heatmap reward
sc = axs[0, 0].scatter(x, y, c=r, cmap='viridis', s=50, edgecolor='k')
fig.colorbar(sc, ax=axs[0, 0], shrink=1.0)
axs[0, 0].set_title('Dataset ($a\sim\pi_b$)')
axs[0, 0].set_xlabel('$s$')
axs[0, 0].set_ylabel('$a$')
# axs[0].set_xlim(0, 4*np.pi)

# Plot sine wave
# axs[1].scatter(x, y, alpha=0.1, color='gray', label='Ground Truth')
# axs[0, 1].scatter(x[0:test_sample], pred_y_gpdq, color='blue', label='GPDQ(RBF)')
sc1 = axs[0, 1].scatter(x[0:test_sample], pred_y_gpdq, c=pred_r_gpdq2, cmap='viridis', s=50, edgecolor='k')
fig.colorbar(sc1, ax=axs[0, 1], shrink=1.0)
axs[0, 1].set_title('$a\sim\pi_{\\boldsymbol{\\theta}}(a^{0:N}|s)$')
axs[0, 1].set_xlabel('$s$')
axs[0, 1].set_ylabel('$a$')
# axs[1].legend()
# axs[1].set_xlim(0, 4*np.pi)

# Plot GP predictions
# axs[2].scatter(x, y, alpha=1, color='gray', label='Ground Truth')
axs[0, 2].plot(x[0:test_sample], pred_g_mu_gpdq, color='blue', label='GPDQ(RBF)')
axs[0, 2].fill_between(x[0:test_sample].flatten(), pred_g_mu_gpdq + pred_g_std_gpdq, pred_g_mu_gpdq - pred_g_std_gpdq, alpha=0.2, color='blue')
axs[0, 2].set_title('$y\sim g(y|\mu_\\boldsymbol{\\omega},\sigma_\\boldsymbol{\\omega})$')
axs[0, 2].set_xlabel('$s$')
axs[0, 2].set_ylabel('$a$')
# axs[2].legend()
# axs[2].set_xlim(0, 4*np.pi)

# Plot kernel matrix
khm = axs[0, 3].imshow(k_nn_gpdq, cmap='plasma', origin='lower')
fig.colorbar(khm, ax=axs[0, 3], shrink=1.0)
axs[0, 3].set_title('$\mathbf{K}(s,s\')$')
# axs[3].set_xlim(0, len(k_nn_gpdq))
# Label x-axis only once
# fig.supxlabel('state (s)')
# fig.supylabel('action (a)')
# plt.tight_layout()

# Plot heatmap reward
sc2 = axs[1, 0].scatter(x, y, c=r, cmap='viridis', s=50, edgecolor='k')
fig.colorbar(sc, ax=axs[1, 0], shrink=1.0)
axs[1, 0].set_title('Dataset ($a\sim\pi_b$)')
axs[1, 0].set_xlabel('$s$')
axs[1, 0].set_ylabel('$a$')
# axs[0].set_xlim(0, 4*np.pi)

# Plot sine wave
# axs[1].scatter(x, y, alpha=0.1, color='gray', label='Ground Truth')
axs[1, 1].scatter(x[0:test_sample], pred_y_gpdq2, color='blue', label='GPDQ(RBF)')
sc3 = axs[1, 1].scatter(x[0:test_sample], pred_y_gpdq2, c=pred_r_gpdq2, cmap='viridis', s=50, edgecolor='k')
fig.colorbar(sc3, ax=axs[1, 1], shrink=1.0)
axs[1, 1].set_title('$a\sim\pi_{\\boldsymbol{\\theta}}(a^{0:N}|s)$')
axs[1, 1].set_xlabel('$s$')
axs[1, 1].set_ylabel('$a$')

# Plot GP predictions
# axs[2].scatter(x, y, alpha=1, color='gray', label='Ground Truth')
axs[1, 2].plot(x[0:test_sample], pred_g_mu_gpdq2, color='blue', label='GPDQ(RBF)')
axs[1, 2].fill_between(x[0:test_sample].flatten(), pred_g_mu_gpdq2 + pred_g_std_gpdq2, pred_g_mu_gpdq2 - pred_g_std_gpdq2, alpha=0.2, color='blue')
axs[1, 2].set_title('$y\sim g(y|\mu_\\boldsymbol{\\omega},\sigma_\\boldsymbol{\\omega})$')
axs[1, 2].set_xlabel('$s$')
axs[1, 2].set_ylabel('$a$')
# axs[2].legend()
# axs[2].set_xlim(0, 4*np.pi)

# Plot kernel matrix
khm2 = axs[1, 3].imshow(k_nn_gpdq2, cmap='plasma', origin='lower')
fig.colorbar(khm2, ax=axs[1, 3], shrink=1.0)
axs[1, 3].set_title('$\mathbf{K}(s,s\')$')

# fig.supxlabel('state (s)')
# fig.supylabel('action (a)')
plt.savefig('sine_2.png', dpi=300, bbox_inches='tight')
plt.show()