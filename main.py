import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import argparse
import os

from GPDQ.GPDQ import GaussianProcessDiffusionQlearning
from IQL.IQL import MyCustomIQL

def addArguments(parser):
    parser.add_argument('--env', default='sharp_sine', help='Environment: ["simple_sine", "sharp_sine"]')
    parser.add_argument('--alg', default="GPDQ", help='Algorithm: ["GPDQ"], default:GPDQ')
    parser.add_argument('--task', default='testing', help='Task: ["training", "testing"], default:training')
    parser.add_argument('--gradient_step', default=50000, help='Gradient Step, default:1000000')

def setGlobalSeed(seed:int):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

def getParamsDict(env):
    return {'simple_sine' : {'environment': 'simple_sine', 'horizon': (4*np.pi//0.05)//2, 'gp_num_sample': 250, 'gp_num_inducing': 25, 'gp_batch_size': 25, 'state_dim': 1, 'action_dim': 1, 'normalise_reward': False, 'diffusion_step': 50, 'task': TASK}, 
            'sharp_sine' : {'environment': 'sharp_sine', 'horizon': (4*np.pi//0.05)//2, 'gp_num_sample': 250, 'gp_num_inducing': 25, 'gp_batch_size': 25, 'state_dim': 1, 'action_dim': 1, 'normalise_reward': False, 'diffusion_step': 50, 'task': TASK}}

def createDataset():
    # Piecewise waveform function
    def non_smooth_function(x):
        return np.piecewise(x,
                            [x < 3, (x >= 3) & (x < 6), x >= 6],
                            [lambda x: np.sin(2 * x),
                            lambda x: np.sin(10 * x),
                            lambda x: np.sin(2 * x + 5)])

    def _getQvalue(x_test):
        _gamma = 0.99
        _q = np.zeros([len(x_test)])
        for i in range(len(_q)):
            for j in range(len(_q)):
                k = i
                if j+k >= len(_q):
                    k = 0
                _q[i] += ((_gamma)**j) * np.abs(np.sin(x_test[j+k]))
        return _q
    
    _x = []
    for _ in range(20):
        _x.append(np.arange(start=0., stop=2*np.pi, step=0.05))
    _x = np.hstack(_x)
    _y = np.zeros([len(_x)])
    _r = np.zeros([len(_y)])  

    for i in range(len(_y)):
        if args.env == 'simple_sine':
            if i <= len(_y)//2:
                _y[i] = np.sin(_x[i]) + np.random.normal(0, 0.05, size=1)
                # _y[i] = non_smooth_function(_x[i]) + np.random.normal(0, 0.05, size=1)
            else:
                _y[i] = -np.sin(_x[i]) + np.random.normal(0, 0.05, size=1)
        elif args.env == 'sharp_sine':
            if i <= len(_y)//2:
                _y[i] = non_smooth_function(_x[i]) + np.random.normal(0, 0.05, size=1)
            else:
                _y[i] = -non_smooth_function(_x[i]) + np.random.normal(0, 0.05, size=1)

    for j in range(len(_r)):
        if _x[j] <= 2*np.pi:
            _r[j] = _y[j]
        else: 
            _r[j] = -_y[j]
    
    _q = _getQvalue(_x[0:250])
        
    _x_p_1 = np.zeros([len(_x)])
    _x_p_1[0:-2] = _x[1:-1]
    _x_p_1[-1] = _x[0]

    return {'arr_0': len(_x),
           'observations': _x.reshape(-1, 1), 
           'actions': _y.reshape(-1, 1), 
           'rewards': _r.reshape(-1, 1), 
           'next_observations': _x_p_1.reshape(-1, 1), 
           'true_q': _q.reshape(-1, 1)}

# ========================================================== Main Program ========================================================== #
seed = 2203
setGlobalSeed(seed)

parser = argparse.ArgumentParser(description='Test Q-value Function')
addArguments(parser)
args = parser.parse_args()

# ========== Global constants
ENV_NAME = args.env
TASK = args.task
EXPERT_PATH = 'datasets/dataset_{}.npz'.format(ENV_NAME)

if not os.path.isfile(EXPERT_PATH):
    dataset = createDataset()
    np.savez(EXPERT_PATH, **dataset)
dataset = np.load(EXPERT_PATH, allow_pickle=True)  # ========== Load the dataset
x = dataset['observations']
y = dataset['actions']
r = dataset['rewards']
q = dataset['true_q']
params_dict = getParamsDict(env=ENV_NAME)

if args.alg == 'GPDQ':
    agent = GaussianProcessDiffusionQlearning(params_dict=params_dict[args.env], dataset=dataset, ft=False)
elif args.alg == 'IQL':
    agent = MyCustomIQL(params_dict=params_dict[args.env], dataset=dataset, ft=False)

# ========== Training (Offline)
EPOCH = int(args.gradient_step) / (agent.NUM_SAMPLE//agent.MINIBATCH_SIZE)
if args.task == 'training':
    # Train Diffusion Policy and Value functions.
    # while agent.beh_training_record < EPOCH or agent.training_record < EPOCH:
    while agent.training_record < EPOCH:
        print('Start Offline Training...')
        # Train the algorithm.
        agent.training(total_epoch=EPOCH, eval=False)
    
    print('Training is done!')
    # pred_y = agent.predict(state=x, size=len(x), guide=False).detach().numpy()
    pred_q = agent.q_1(torch.concat([torch.tensor(x[0:250], dtype=torch.float32), 
                                     torch.tensor(y[0:250], dtype=torch.float32)], dim=1)).tolist()
if args.task == 'testing':
    pred_y = agent.predict(state=x[0:250], size=250, guide=True).detach().numpy()
    pred_q = agent.q_1(torch.concat([torch.tensor(x[0:250], dtype=torch.float32), 
                                     torch.tensor(pred_y, dtype=torch.float32)], dim=1)).tolist()
    np.save(agent.q_eval_path, pred_q)

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
# Plot sine wave
axs[0].scatter(x, y, color='blue', label='Ground Truth')
axs[0].scatter(x[0:250], pred_y, color='purple', label='GPDQ')
axs[0].set_title('Sine Wave')
axs[0].grid(True)
axs[0].legend()
# Plot negative sine wave
axs[1].scatter(x[0:250], q, color='black', label='Ground Truth (Q)')
# axs[1].scatter(x, r, color='blue', label='Ground Truth (R)')
axs[1].scatter(x[0:250], pred_q, color='blue', label='Q-GPDQ')
# axs[1].scatter(x, r, color='red', label='Rewards')
axs[1].set_title('State-Action Value Function (Q-value)')
axs[1].grid(True)
axs[1].legend()

# Label x-axis only once
plt.xlabel('x')
plt.tight_layout()
plt.show()