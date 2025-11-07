import os
import torch
import numpy as np
from time import time

class mySVGP(torch.nn.Module):
    def __init__(self, parent_alg:str, params_dict:dict, dataset:dict, cuda:bool=False):
        super().__init__()
        self.parent_alg = parent_alg
        self.alg = 'svgp'
        self.param_dict = params_dict
        self.env = params_dict['environment']
        # self.num_sample = 10000  # Fixed to 1000.
        self.num_sample = dataset['arr_0']
        # self.num_sample = params_dict['gp_num_sample']
        # self.num_inducing = 1000
        self.num_inducing = params_dict['gp_num_inducing']
        self.x_dim = int(params_dict['state_dim'])
        self.y_dim = int(params_dict['action_dim'])
        # self.training_record_path = '{}/training_records/{}_{}_{}_training_records_{:d}_{:d}.zip'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.num_inducing)
        
        # self.training_record_path = '{}/training_records/walker2d_1st_(rbf-1000-3hidden)/{}_{}_{}_training_records_{:d}_{:d}.zip'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.num_inducing)
        # self.training_record_path = '{}/training_records/hopper_1st_(rbf-1000-3hidden)/{}_{}_{}_training_records_{:d}_{:d}.zip'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.num_inducing)
        # self.training_record_path = '{}/training_records/halfcheetah_1st_(rbf-1000-3hidden)/{}_{}_{}_training_records_{:d}_{:d}.zip'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.num_inducing)
        self.training_record_path = '{}/training_records/halfcheetah_2nd_(svm-1000-3hidden)/{}_{}_{}_training_records_{:d}_{:d}.zip'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.num_inducing)

        self.training_checkpoint_path = '{}/training_records/{}_{}_{}_checkpoint_{:d}_{:d}.zip'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.num_inducing)
        self.cuda() if cuda else ...

        # Check for the training record file and the response from user...
        if os.path.isfile(self.training_record_path) is True:
        # if os.path.isfile(self.training_checkpoint_path) is True:
            print('========== ({:s}) There exists a training record for this agent. Do you wish to load the exist one ?'.format(self.alg))
            _ans_1 = input('========== ({:s}) Press [y/n] and enter: '.format(self.alg))
        else:
            _ans_1 = 'n'
        # Check for the answer...
        if _ans_1 == 'n' or _ans_1=='N' or _ans_1=='No' or _ans_1=='NO':
            print('========== ({:s}) Create new GP record and models!'.format(self.alg))
            self.mll_append = []  # Loss storage
            self.training_record = 0  # Training record counter
            # Initialise training data
            _start = 0
            self.x_train = torch.tensor(dataset['observations'][0+_start:_start+self.num_sample], dtype=torch.float32)
            self.y_train = torch.tensor(dataset['actions'][0+_start:_start+self.num_sample], dtype=torch.float32)
            # Intialise inducing points (Let's fix it first.)
            # _best_z, _best_z_indexes = self.getInitInducingPoints(x_train=self.x_train[0:1000])
            # _sub_z, _sub_z_indexes = self.getInitInducingPoints(x_train=self.x_train[60000:61000])
            # self.z_train = torch.concatenate([_best_z, _sub_z], dim=0)
            # self.indexes = np.concatenate([_best_z_indexes, _sub_z_indexes], axis=0)
            self.z_train, self.indexes = self.getInitInducingPoints(x_train=self.x_train[0:1000])
            # Create hyperparameters
            self.sigma_n = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
            self.ell = torch.nn.Parameter(torch.ones(size=[1, self.x_dim], dtype=torch.float32), requires_grad=True)

            self.q_mean = torch.nn.Parameter(self.y_train[self.indexes], requires_grad=True)
            self.q_var = torch.nn.Parameter(torch.ones([self.num_inducing, self.num_inducing]), requires_grad=True)  # Single covariance
            self.z_train = torch.nn.Parameter(self.z_train, requires_grad=True)

            self.num_mixtures = params_dict['state_dim']
            self.mean_q = torch.nn.Parameter(torch.zeros(self.num_mixtures, self.x_dim), requires_grad=True)
            self.weight_q = torch.nn.Parameter(torch.ones(self.num_mixtures), requires_grad=True)
            self.v_q = torch.nn.Parameter(torch.ones(self.num_mixtures, self.x_dim), requires_grad=True)

        elif _ans_1 == 'y' or _ans_1=='Y' or _ans_1=='Yes' or _ans_1=='YES':
            print('========== ({:s}) Load training record.'.format(self.alg))
            # Load training records
            # _training_records = torch.load(self.training_record_path, map_location=torch.device('cuda' if cuda else 'cpu'))
            # # _training_records = torch.load(self.training_checkpoint_path, map_location=torch.device('cuda' if cuda else 'cpu'))
            # self.mll_append = _training_records['loss_append']  # Loss append
            # self.training_record = _training_records['training_record']  # Training record counter
            # _state_dict = _training_records['state_dict']
            # self.x_train = _training_records['x_train']
            # self.y_train = _training_records['y_train']
            # self.z_train = torch.nn.Parameter(_training_records['z_train'], requires_grad=True)
            # self.sigma_n = torch.nn.Parameter(_state_dict['sigma_n'], requires_grad=True)
            # self.ell = torch.nn.Parameter(_state_dict['ell'], requires_grad=True)
            # self.q_mean = torch.nn.Parameter(_state_dict['q_mean'], requires_grad=True)
            # self.q_var = torch.nn.Parameter(_state_dict['q_var'], requires_grad=True)

            self.loadTrainingRecord(path=self.training_record_path, cuda=cuda)
            # self.loadTrainingRecord(path=self.training_checkpoint_path, cuda=cuda)
            self.eval()
        else:
            print('========== ({:s}) Try another answer. ("y" for yes (create new) or "n" for no (load existing one)).'.format(self.alg))
            exit()

        # Summarize parameters
        self.sigma_p = torch.tensor(1.0, dtype=torch.float32)  # Fixed signal variance to 1.00^2
        # print('========== ({:s}) Summarize GP parameters.'.format(self.alg))
        # print('sigma_p: ', self.sigma_p.tolist(), ', sigma_n: ', self.sigma_n.tolist(), ', ell: ', self.ell.tolist())
        # print('q_mean: ', self.q_mean.tolist(), ', q_var: ', torch.mean(torch.diag(self.q_var)).tolist(), ', z_size: ', self.z_train.size())

        # Initialised training kernels (K, K_inv) (For evaluation).
        self.K_mm = self.rbfKernel(X_1=self.z_train, X_2=self.z_train, noise=True)
        self.L_mm = torch.linalg.cholesky(self.K_mm, upper=False)

    def loadTrainingRecord(self, path:str, cuda:bool):
        _training_records = torch.load(path, map_location=torch.device('cuda' if cuda else 'cpu'))
        # _training_records = torch.load(self.training_checkpoint_path, map_location=torch.device('cuda' if cuda else 'cpu'))
        self.mll_append = _training_records['loss_append']  # Loss append
        self.training_record = _training_records['training_record']  # Training record counter
        _state_dict = _training_records['state_dict']
        self.x_train = _training_records['x_train']
        self.y_train = _training_records['y_train']
        
        self.sigma_n = torch.nn.Parameter(_state_dict['sigma_n'], requires_grad=True)
        self.ell = torch.nn.Parameter(_state_dict['ell'], requires_grad=True)

        self.q_mean = torch.nn.Parameter(_state_dict['q_mean'], requires_grad=True)
        self.q_var = torch.nn.Parameter(_state_dict['q_var'], requires_grad=True)
        self.z_train = torch.nn.Parameter(_training_records['z_train'], requires_grad=True)

        self.num_mixtures = self.param_dict['state_dim']
        self.mean_q = torch.nn.Parameter(_state_dict['mean_q'], requires_grad=True)
        self.weight_q = torch.nn.Parameter(_state_dict['weight_q'], requires_grad=True)
        self.v_q = torch.nn.Parameter(_state_dict['v_q'], requires_grad=True)

    def getInitInducingPoints(self, x_train):
        _indexes = np.linspace(start=0, stop=len(x_train), num=self.num_inducing, endpoint=False)
        return x_train[_indexes], _indexes
    
    def rbfKernel(self, X_1, X_2, noise=False):
        '''
        Isotropic squared exponential kernel.
        Args:
            X1: Array of m points (m x d).
            X2: Array of n points (n x d).
        Returns:
            (m x n) matrix.
        '''
        # # RBF
        # X_1 = torch.tensor(X_1, dtype=torch.float32) if not torch.is_tensor(X_1) else X_1
        # X_2 = torch.tensor(X_2, dtype=torch.float32) if not torch.is_tensor(X_2) else X_2
        # # kernel = (self.sigma_p**2) * torch.exp(-(torch.cdist(X_1, X_2)**2)/(2*self.ell**2))
        # kernel = (self.sigma_p**2) * torch.exp(-(torch.cdist(X_1/self.ell, X_2/self.ell)**2)/2)
        # if noise:
        #     # Noisy observation
        #     kernel += ((self.sigma_n**2) * torch.eye(len(X_1)))

        # return kernel

        # # SVM
        # X_1 = torch.tensor(X_1, dtype=torch.float32) if not torch.is_tensor(X_1) else X_1
        # X_2 = torch.tensor(X_2, dtype=torch.float32) if not torch.is_tensor(X_2) else X_2     
        kernel = 0
        for i in range(self.num_mixtures):
            _diff = torch.cdist(X_1*self.v_q[i], X_2*self.v_q[i])
            _exp_term = torch.exp(-2 * (torch.pi**2) * (_diff**2))
            _cos_term = torch.cos(2 * torch.pi * torch.cdist(X_1*self.mean_q[i], X_2*self.mean_q[i]))
            kernel += self.weight_q[i] * (_exp_term * _cos_term)
        if noise:
            # Noisy observation
            kernel += ((self.sigma_n**2) * torch.eye(len(X_1)))
        return kernel  

    def predict(self, X_s):
        with torch.no_grad():
            x_test = X_s.view(-1, self.x_dim)
            _k_s = self.rbfKernel(X_1=self.z_train, X_2=x_test, noise=False)
            _k_ss = self.rbfKernel(X_1=x_test, X_2=x_test, noise=False)

            # # We found that deriving mean and variance this way, provide better results.
            # _mean = _k_s.T @ self.K_inv @ self.y_train
            # _var = _k_ss - _k_s.T @ self.K_inv @ _k_s

            # # Cholesky decomposition
            # _L = torch.linalg.cholesky(self.K, upper=False)
            # _alpha = torch.linalg.solve_triangular(_L.T, torch.linalg.solve_triangular(_L, self.y_train, upper=False), upper=True)
            # _mean = _k_s.T @ _alpha

            # _mean = _k_s.T @ (torch.linalg.solve_triangular(self.L_mm.T, torch.linalg.solve_triangular(self.L_mm, self.y_train, upper=False), upper=True))
            _mean = _k_s.T @ (torch.linalg.solve_triangular(self.L_mm.T, torch.linalg.solve_triangular(self.L_mm, self.q_mean, upper=False), upper=True))
            _v = torch.linalg.solve_triangular(self.L_mm, _k_s, upper=False)
            _k_tilde = _k_ss - (_v.T @ _v)
            _1 = torch.linalg.solve_triangular(self.L_mm.T, torch.linalg.solve_triangular(self.L_mm, self.q_var, upper=False), upper=True)
            _2 = torch.linalg.solve_triangular(self.L_mm.T, torch.linalg.solve_triangular(self.L_mm, _k_s, upper=False), upper=True)
            _sigma_f = _k_tilde + (_k_s.T @ _1 @ _2)
            _var = (_sigma_f) + (self.sigma_n**2)

        return _mean, _var

    # Training Method for GPR. 
    # The reason we train GP here is to add entropy term to regulate the covariance. 
    def myTraining(self, total_epoch:int, ft:bool=False):
        _batch_size = self.param_dict['gp_batch_size']
        _gradient_step = int(self.num_sample//_batch_size)  
        # _cov_optimizer = torch.optim.Adam([self.sigma_n, self.ell], lr=1e-02)
        _cov_optimizer = torch.optim.Adam([self.sigma_n, self.mean_q, self.v_q, self.weight_q], lr=3e-04)
        _ind_optimizer = torch.optim.Adam([self.q_mean, self.q_var, self.z_train], lr=1e-04)  # was 1e-04
        self.train()  # Set to training mode.
        _training_record = 0

        while _training_record < total_epoch:
        # while self.training_record < total_epoch:
            _start_time = time()
            _sampling_indexes = torch.randperm(self.num_sample)
            for g in range(_gradient_step):
                _batch_x = self.x_train[_sampling_indexes[g*_batch_size:_batch_size+(g*_batch_size)]]
                _batch_y = self.y_train[_sampling_indexes[g*_batch_size:_batch_size+(g*_batch_size)]]
                _batch_z = self.z_train
                _l_1 = 0
                _cov_optimizer.zero_grad()
                _ind_optimizer.zero_grad()
                _K_mm = self.rbfKernel(X_1=_batch_z, X_2=_batch_z, noise=True)
                _K_nn = self.rbfKernel(X_1=_batch_x, X_2=_batch_x, noise=True)
                _L_mm = torch.linalg.cholesky(_K_mm, upper=False)
                _K_mn = self.rbfKernel(X_1=_batch_z, X_2=_batch_x, noise=False)
                _K_tilde = _K_nn - _K_mn.T @ (torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, _K_mn, upper=False), upper=True))

                _1 = torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, self.q_var, upper=False), upper=True)
                _3 = torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, self.q_mean, upper=False), upper=True)
                _sigma_n_sq = self.sigma_n**2
                for h in range(_batch_size-1):
                    _k_i = _K_mn[:, h].view(-1, 1)
                    _mean = _k_i.T @ _3
                    _2 = torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, _k_i, upper=False), upper=True)
                    _sigma_f_sq = torch.abs(_K_tilde[h, h].view(1, 1) + (_k_i.T @ _1 @ _2))
                    _l_1 += (-(_batch_y[h].view(1, self.y_dim)-_mean)**2) / (2*_sigma_n_sq) - \
                            (torch.log(torch.sqrt(2*torch.pi*_sigma_n_sq))) - \
                            (_sigma_f_sq / (2*_sigma_n_sq))

                # KL-divergence term
                _cov_q = torch.sqrt(torch.abs(torch.diagonal(self.q_var))) @ torch.ones(size=[self.num_inducing, self.y_dim])
                _q = torch.distributions.Normal(loc=self.q_mean, scale=_cov_q)
                _u = _q.sample(torch.Size())
                _log_q_u = _q.log_prob(_u)
                _p = torch.distributions.Normal(loc=0., 
                                                scale=torch.sqrt(torch.abs(torch.diagonal(_K_mm)))@torch.ones(size=[self.num_inducing, self.y_dim]))

                # ELBO
                _elbo = _l_1 - (-torch.sum(torch.exp(_log_q_u)*(_p.log_prob(_u)-_log_q_u)))
                _elbo = -_elbo.mean()  # We maximise ELBO.
                _elbo.backward(retain_graph=True)
                _cov_optimizer.step()
                _ind_optimizer.step()

                self.mll_append.append(_elbo.tolist())

            self.training_record += 1
            _training_record += 1

            # print('Gradient_Step: ', self.training_record, 
            #         ', Loss: ', round(_elbo.tolist(), 4), 
            #         ', Var: ', round(torch.diagonal(self.q_var).mean().tolist(), 4),
            #         ', Obs_var: ', round(self.sigma_n.tolist(), 4),
            #         ', Time_usage: ', round(time()-_start_time, 4))

        self.recordSaving(path=self.training_record_path)

        # self.eval()  # Set to evaluation mode after the training is finished.
        self.K_mm = self.rbfKernel(X_1=self.z_train, X_2=self.z_train, noise=True)
        self.L_mm = torch.linalg.cholesky(self.K_mm, upper=False)

        return _elbo.tolist()

    def recordSaving(self, path:str):
        torch.save({'state_dict': self.state_dict(), 
                    'training_record': self.training_record,
                    'loss_append': self.mll_append,
                    'x_train': self.x_train, 
                    'y_train': self.y_train,
                    'z_train': self.z_train}, path)