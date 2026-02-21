import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from scipy.interpolate import griddata
# sns.set_theme()

def runningMeanAndStd(data):
    _num_convo = 10
    _r_mean = np.zeros([len(data)])
    _r_std = np.zeros([len(data)])
    for j in range(len(data)):
        _r_mean[j] = np.mean(data[0+np.abs(j-_num_convo//2):j+1+_num_convo//2])
        _r_std[j] = np.std(data[0+np.abs(j-_num_convo//2):j+1+_num_convo//2])
    return _r_mean, _r_std

EXPERT_PATH = 'datasets/dataset_simple_sine.npz'
# EXPERT_PATH = 'datasets/dataset_sharp_sine.npz'
test_sample = 628

dataset = np.load(EXPERT_PATH, allow_pickle=True)  # ========== Load the dataset
x = dataset['observations'].ravel()
y = dataset['actions'].ravel()
r = dataset['rewards'].ravel()
q = dataset['true_q']

points = np.column_stack((x, y)) 
X, Y = np.meshgrid(
    np.linspace(0, x.max(), test_sample),
    np.linspace(y.min(), y.max(), test_sample)
)
R = griddata((x, y), r, (X, Y), method='cubic')
R_nearest = griddata(points, r, (X, Y), method='nearest')
# Fill NaNs from cubic with nearest
R = np.where(np.isnan(R), R_nearest, R)

# pi_b data
y_mean, y_std = runningMeanAndStd(y[0:test_sample])

# Load test data
# GPDQ (RBF)
gpdq_test_data = np.load('GPDQ_rbf/evaluation_data.npz')
pred_y_gpdq = gpdq_test_data['arr_0']
pred_g_mu_gpdq = gpdq_test_data['arr_1']
pred_g_var_gpdq = gpdq_test_data['arr_2']
pred_g_std_gpdq = np.sqrt(np.diagonal(pred_g_var_gpdq))
pred_q_gpdq = gpdq_test_data['arr_3']
k_nn_gpdq = gpdq_test_data['arr_4']
pred_y_mean, pred_y_std = runningMeanAndStd(pred_y_gpdq)

# DQL
dql_test_data = np.load('DQL/evaluation_data.npz')
pred_y_dql = dql_test_data['arr_0']
pred_q_dql = dql_test_data['arr_3']
pred_y_dql_mean, pred_y_dql_std = runningMeanAndStd(pred_y_dql)

# Plot sine wave
plt.scatter(x, y, alpha=0.5, color='gray', label='$\\boldsymbol{\pi}_b$')
bg = plt.imshow(R, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto', cmap='viridis', alpha=0.7)
plt.colorbar(bg, label="Reward")
# plt.plot(x[0:test_sample], y[0:test_sample], alpha=1, color='orange', label='best of $\\boldsymbol{\pi}_b$')
# plt.plot(x[0:test_sample], pred_y_gpdq, color='red', label='$\\boldsymbol{\pi}_\\theta$(Ours)')

# plt.scatter(x[0:test_sample], y[0:test_sample], alpha=1, color='red', label='$\pi_b$')
# plt.scatter(x[0:test_sample], pred_y_gpdq, color='blue', label='$\pi_\\theta$(Ours)')

plt.plot(x[0:test_sample], y_mean, alpha=1, color='orange', label='best of $\\boldsymbol{\pi}_b$')
plt.fill_between(x[0:test_sample], y_mean+y_std, y_mean-y_std, color='orange', alpha=0.2)

plt.plot(x[0:test_sample], pred_y_mean, alpha=1, color='red', label='$\\boldsymbol{\pi}_\\theta$(Ours)')
plt.fill_between(x[0:test_sample], pred_y_mean+pred_y_std, pred_y_mean-pred_y_std, color='red', alpha=0.2)

# plt.plot(x[0:test_sample], pred_y_dql_mean, alpha=1, color='purple', label='$\\boldsymbol{\pi}_\\theta$(DQL)')
# plt.fill_between(x[0:test_sample], pred_y_dql_mean+pred_y_dql_std, pred_y_dql_mean-pred_y_dql_std, color='purple', alpha=0.2)

# plt.title('Policy Prediction on Sine-wave data')
plt.xlabel('State')
plt.ylabel('Action')
# plt.legend()
plt.tight_layout()
plt.savefig('sine_1.png', dpi=300, bbox_inches='tight')
# plt.show()