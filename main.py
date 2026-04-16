import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import argparse
import os

from GPDQ_rbf.GPDQ_rbf import GaussianProcessDiffusionQlearning
# from GPDQ_matern2.GPDQ_matern2 import GaussianProcessDiffusionQlearning
# from GPDQ_matern.GPDQ_matern import GaussianProcessDiffusionQlearning
# from GPDQ_matern52.GPDQ_matern52 import GaussianProcessDiffusionQlearning
# from GPDQ_Resnet.GPDQ_Resnet import GaussianProcessDiffusionQlearning
# from GPDQ_dkl.GPDQ_dkl import GaussianProcessDiffusionQlearning
# from GPDQ_NZM_matern.GPDQ_NZM_matern import GaussianProcessDiffusionQlearning
from IQL.IQL import MyCustomIQL
from DQL.DQL import MyCustomDQL

def addArguments(parser):
    parser.add_argument('--env', default='simple_sine', help='Environment: ["simple_sine", "sharp_sine"]')
    parser.add_argument('--alg', default="GPDQ", help='Algorithm: ["GPDQ"], default:GPDQ')
    parser.add_argument('--task', default='training', help='Task: ["training", "testing"], default:training')
    parser.add_argument('--gradient_step', default=10000, help='Gradient Step, default:1000000')

def setGlobalSeed(seed:int):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

def getParamsDict(env):
    return {'simple_sine' : {'environment': 'simple_sine', 'horizon': int(4*np.pi//0.02), 'gp_num_sample': int(1.5*np.pi//0.02), 'gp_num_inducing': 25, 'gp_batch_size': 25, 'state_dim': 1, 'action_dim': 1, 'normalise_reward': False, 'diffusion_step': 50, 'task': TASK}, 
            'sharp_sine' : {'environment': 'sharp_sine', 'horizon': 628, 'gp_num_sample':314, 'gp_num_inducing': 25, 'gp_batch_size': 25, 'state_dim': 1, 'action_dim': 1, 'normalise_reward': False, 'diffusion_step': 50, 'task': TASK}}

def createDataset():

    def fluctuate_function(x):
        if x < 0.5:
            y = 0.3 * np.sin(x)
        elif x >= 0.5 and x < 0.7:
            y = 0.7 * np.sin(x)
        elif x >= 1.5 and x < 2.7:
            y = (0.2 * np.sin(x))
        elif x >= 3.9 and x < 5.1:
            y = (0.85 * np.sin(x))
        elif x >= 5.6:
            y = 0.4 * np.sin(x)
        else:
            y = np.sin(x)
        return y
    def _getQvalue(x_test, r_test):
        _gamma = 0.99
        _q = np.zeros([len(x_test)])
        for i in range(len(_q)):
            for j in range(len(_q)):
                k = i
                if j + k >= params_dict[args.env]['horizon']:
                    k = 0
                # _q[i] += ((_gamma) ** j) * (np.sin(x_test[j+k]) + non_smooth_function(x_test[j+k]))
                _q[i] += ((_gamma) ** j) * np.abs(r_test[j+k])
        return _q

    _x = []
    for _ in range(100):
        _x.append(np.arange(start=0., stop=4*np.pi, step=0.02))
    _x = np.hstack(_x)
    _y = np.zeros([len(_x)])
    _r = np.zeros([len(_y)])

    for i in range(len(_y)):
        if args.env == 'simple_sine':
            if _x[i] < 1.5*np.pi:
                _mul = 1
                _mux = 1
            elif _x[i] >= 1.5*np.pi and _x[i] < 2.75*np.pi:
                _mul = 1
                _mux = 1
            elif _x[i] >= 2.75*np.pi and _x[i] < 3.5*np.pi:
                _mul = 1
                _mux = 1
            else:
                _mul = 1
                _mux = 1
            if i <= len(_y) // 2:
                _y[i] = _mux * np.sin(_mul*_x[i]) + np.random.normal(0, 0.05, size=1)
                # _y[i] = non_smooth_function(_x[i]) + np.random.normal(0, 0.05, size=1)
            else:
                _y[i] = _mux * (-np.sin(_mul*_x[i]) + np.random.normal(0, 0.05, size=1))
        elif args.env == 'sharp_sine':
            if i <= len(_y) // 2:
                _y[i] = fluctuate_function(_x[i]) + np.random.normal(0, 0.05, size=1)
                # _y[i] = np.sin(3 * _x[i]) + np.random.normal(0, 1, size=1)
            else:
                _y[i] = -fluctuate_function(_x[i]) + np.random.normal(0, 0.05, size=1)
                # _y[i] = -np.sin(3 * _x[i]) + np.random.normal(0, 1, size=1)

            _y[i] = np.clip(_y[i], -1, 1)

    for j in range(len(_r)):
        if _x[j] <= np.pi or _x[j] >= 3 * np.pi:
            _r[j] = _y[j]
        else:
            _r[j] = -_y[j]

    _q = _getQvalue(_x[0:628], _r[0:628])
    # _q = _getQvalue(_x[0:314], _r[0:314])

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
params_dict = getParamsDict(env=ENV_NAME)

if not os.path.isfile(EXPERT_PATH):
    dataset = createDataset()
    np.savez(EXPERT_PATH, **dataset)
dataset = np.load(EXPERT_PATH, allow_pickle=True)  # ========== Load the dataset
x = dataset['observations']
y = dataset['actions']
r = dataset['rewards']
q = dataset['true_q']


if args.alg == 'GPDQ':
    agent = GaussianProcessDiffusionQlearning(params_dict=params_dict[args.env], dataset=dataset, ft=False)
    # test_sample = agent.gp_model.num_sample
    test_sample = params_dict[args.env]['horizon']
    # test_sample = 500
    # dp_test_sample = 314
elif args.alg == 'IQL':
    agent = MyCustomIQL(params_dict=params_dict[args.env], dataset=dataset, ft=False)
    test_sample = params_dict[args.env]['horizon']
elif args.alg == 'DQL':
    agent = MyCustomDQL(params_dict=params_dict[args.env], dataset=dataset, ft=False)
    test_sample = params_dict[args.env]['horizon']

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
    pred_q = agent.q_1(torch.concat([torch.tensor(x[0:agent.gp_model.num_sample], dtype=torch.float32),
                                     torch.tensor(y[0:agent.gp_model.num_sample], dtype=torch.float32)], dim=1)).tolist()
if args.task == 'testing':
    if agent.ALG == 'IQL':
        pred_y = np.abs(y[0:test_sample])
        for k in range(len(x[0:test_sample])):
            if x[k] > np.pi and x[k] < 3*np.pi:
                pred_y[k] = -pred_y[k]
        pred_g_mu = np.zeros([test_sample], dtype=float)
        pred_g_var = np.ones([test_sample], dtype=float)
        k = np.identity(n=params_dict[args.env]['gp_num_sample'])
    elif agent.ALG == 'DQL':
        pred_y = agent.predict(state=x[0:test_sample], size=test_sample, guide=True).cpu().detach().numpy()
        pred_g_mu = np.zeros([test_sample], dtype=float)
        pred_g_var = np.ones([test_sample], dtype=float)
        k = np.identity(n=params_dict[args.env]['gp_num_sample'])
    else:
        pred_y = agent.predict(state=x[0:test_sample], size=test_sample, guide=True).cpu().detach().numpy()
        pred_g_mu, pred_g_var = agent.predictGP(torch.tensor(x[0:test_sample], dtype=torch.float32).view(-1, 1))
        pred_g_mu = np.reshape(pred_g_mu.tolist(), -1)
        pred_g_var = pred_g_var.detach().cpu().numpy()
        k = agent.gp_model.rbfKernel(X_1=agent.gp_model.x_train, X_2=x[0:test_sample], noise=False).cpu().detach().numpy()
        # pred_g_var = pred_g_var

    pred_q = agent.q_1(torch.concat([torch.tensor(x[0:test_sample], dtype=torch.float32),
                                     torch.tensor(pred_y, dtype=torch.float32)], dim=1)).tolist()
    # np.save(agent.q_eval_path, pred_q)
    np.savez(agent.evaluation_path,
             pred_y,         # arr_0
             pred_g_mu,      # arr_1
             pred_g_var,     # arr_2
             pred_q,         # arr_3
             k)           # arr_4
