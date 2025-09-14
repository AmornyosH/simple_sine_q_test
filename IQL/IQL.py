'''
Gaussian Process Diffusion Policy
Version: v0
Revision: 1
Remark: PyTorch version
'''
# ============================ Pytorch Related ============================
import torch
CUDA = torch.cuda.is_available()
# ============================ Pytorch Related ============================

# ================================== Others ==================================
from my_utilities import my_utils, my_NN
from time import time
import numpy as np
import os

class MyCustomIQL:
    def __init__(self, params_dict:dict, dataset:dict, ft:bool=False):
        self.ALG = 'IQL'
        self.ENV_CONFIG = params_dict['environment']
        self.STATE_DIM = int(params_dict['state_dim'])
        self.ACTION_DIM = int(params_dict['action_dim'])
        
        if dataset is not None:
            self.NUM_SAMPLE = dataset['arr_0']
            # Initialise replay buffers
            self.state_buffer = torch.tensor(dataset['observations'], dtype=torch.float32)        # State buffer (unnormalised)
            self.next_state_buffer = torch.tensor(dataset['next_observations'], dtype=torch.float32)   # Next state buffer (unnormalised)
            self.action_buffer = torch.tensor(dataset['actions'], dtype=torch.float32)       # Action buffer
            self.reward_buffer = torch.tensor(dataset['rewards'], dtype=torch.float32)       # Reward buffer
            self.MINIBATCH_SIZE = 256
        else:
            self.NUM_SAMPLE = 1e+06
            self.MINIBATCH_SIZE = 256

        # Initialise Paths
        self.training_record_path = 'IQL/training_records/{:s}_{:s}_training_record'.format(self.ALG, self.ENV_CONFIG)
        self.training_record_2m_path = 'IQL/training_records/{:s}_{:s}_training_record_2m'.format(self.ALG, self.ENV_CONFIG)
        self.training_checkpoint_path = 'IQL/training_records/{:s}_{:s}_checkpoint'.format(self.ALG, self.ENV_CONFIG)
        self.evaluation_path = 'IQL/norm_eval_rewards_append'
        self.q_eval_path = '{:s}/pred_q'.format(self.ALG)

        # Initialise neural networks
        self.Q_INPUT_DIM = self.STATE_DIM + self.ACTION_DIM
        self.V_INPUT_DIM = self.STATE_DIM
        # Check for the training record file and the response from user...
        if os.path.isfile(self.training_record_path) is True:
            print('========== ({:s}) There exists a training record for this agent. Do you wish to load the exist one ?'.format(self.ALG))
            _ans_1 = input('========== ({:s}) Press [y/n] and enter: '.format(self.ALG))
        else:
            _ans_1 = 'n'
        # Check for the answer...
        if _ans_1 == 'n' or _ans_1=='N' or _ans_1=='No' or _ans_1=='NO':
            print('========== ({:s}) Create new record and models!'.format(self.ALG))
            self.training_record = 0
            self.epsilon_beh_loss_append = []
            self.q_1_loss_append = []
            self.q_2_loss_append = []
            self.q_1 = my_NN.MLP(input_dim=self.Q_INPUT_DIM, output_dim=1)
            self.q_2 = my_NN.MLP(input_dim=self.V_INPUT_DIM, output_dim=1)
            # Initialise Target models
            self.q_1_tar = self.q_1
            self.q_2_tar = self.q_2
        elif _ans_1 == 'y' or _ans_1=='Y' or _ans_1=='Yes' or _ans_1=='YES':
            print('========== ({:s}) Load training record.'.format(self.ALG))
            self.loadTrainingRecord()  # Load from the method here <--------- ****
            self.q_1.eval()
            self.q_2.eval()
        else:
            print('========== ({:s}) Try another answer. ("y" for yes (load existing one) or "n" for no (create new)).'.format(self.ALG))
            exit()
            
        self.norm_reward_training_append = []
        self.best_norm_reward_training = 0
        # Declare optimizer for the networks (offline training)
        self.q_1_optimizer = torch.optim.Adam(self.q_1.parameters(), lr=3e-04)
        self.q_2_optimizer = torch.optim.Adam(self.q_2.parameters(), lr=3e-04)
        # Set to cuda if GPU is available.
        if CUDA:
            self.q_1.cuda()
            self.q_2.cuda()

    # Training Record Loading Method
    def loadTrainingRecord(self):
        _loaded_training_record = torch.load(self.training_record_path, map_location=torch.device('cpu' if not CUDA else 'cuda'), weights_only=False)
        # _loaded_training_record = torch.load(self.training_checkpoint_path, map_location=torch.device('cpu' if not CUDA else 'cuda'))
        self.training_record = _loaded_training_record['training_record']
        self.norm_reward_training_append = _loaded_training_record['norm_reward_training_append']
        self.q_1 = _loaded_training_record['q_1']
        self.q_2 = _loaded_training_record['q_2']
        self.q_1_tar = _loaded_training_record['q_1_tar']
        self.q_2_tar = _loaded_training_record['q_2_tar']
        self.epsilon_beh_loss_append = _loaded_training_record['epsilon_beh_loss_append']
        self.q_1_loss_append = _loaded_training_record['q_1_loss_append']
        self.q_2_loss_append = _loaded_training_record['q_2_loss_append']
        print('========== ({:s}) Training Record: '.format(self.ALG), self.training_record, ' epoch.', 
              ', Gradient steps: ', self.training_record*(self.NUM_SAMPLE//self.MINIBATCH_SIZE))

    # Training Method (for offliine training)
    def training(self, total_epoch:int, eval:bool=False, ft:bool=False):
        # Get Expected Bellman's Equation (local)
        def _getExpectedCumulativeReturn(inputs):
            return batch_reward_tensor + (_GAMMA * self.q_2(inputs))
        
        # Get Expected Q value Method (local)
        def _getExpectedQValues(inputs):
            return self.q_1_tar(inputs)
        
        # Q network Training Method (local)
        def _trainQ1Network(inputs, y_true):
            self.q_1_optimizer.zero_grad()
            y_pred = self.q_1(inputs)
            _q_1_loss = torch.mean(torch.square(y_true - y_pred))

            return _q_1_loss

        # V network Training Method (local)
        def _trainVNetwork(inputs, y_true):
            _TAU = 0.90
            # Compute gradients
            self.q_2_optimizer.zero_grad()
            y_pred_tensor = self.q_2(inputs)
            _td = y_true - y_pred_tensor
            _td_cond = torch.where(_td < 0, 1., 0.)
            _q_2_loss = torch.abs(_TAU - _td_cond) * torch.square(_td)
            _q_2_loss = torch.mean(_q_2_loss)
            return _q_2_loss
        
        # Diffusion Models Training Method (local)
        def _trainDiffusionBeh(inputs, y_true):
            self.epsilon_optimizer.zero_grad()
            residual_noise = self.epsilon_beh(inputs)
            # Compute for the MSE.
            _diffu_loss = torch.mean(torch.square(y_true - residual_noise), dim=1, keepdim=True)
            # Mean of Loss 
            _diffu_loss = torch.mean(_diffu_loss)
            _diffu_loss.backward()
            self.epsilon_optimizer.step()
            return _diffu_loss, _diffu_loss.grad

        # Target Networks Updating Method
        def _updateTargetNetworks():
            # Update the target networks
            q_1_tar_state_dict = self.q_1_tar.state_dict()
            q_1_state_dict = self.q_1.state_dict()
            for key in q_1_state_dict:
                q_1_tar_state_dict[key] = q_1_state_dict[key]*_TAU + q_1_tar_state_dict[key]*(1-_TAU)
            self.q_1_tar.load_state_dict(q_1_tar_state_dict)

        # Declare Constants
        _GAMMA = 0.99  # Discount factor.
        _TAU = 0.005  # Tau for soft updating of target model's weights.

        # Extract dataset
        buffer_size = self.NUM_SAMPLE if not ft else self.state_buffer.size()[0]
        state_buffer = self.state_buffer
        action_buffer = self.action_buffer
        reward_buffer = self.reward_buffer
        next_state_buffer = self.next_state_buffer

        # Initialise parameters
        _batch_size = self.MINIBATCH_SIZE
        _num_gradient_step = buffer_size//_batch_size
        _training_record = self.training_record if not ft else 0

        # Set models to training mode.
        self.q_1.train()
        self.q_2.train()

        # ========================= Training Loop Start =========================
        while _training_record < total_epoch:
            start_time = time()
            diffu_loss_accum = 0
            q_1_loss_accum = 0
            q_2_loss_accum = 0

            # Get shuffle indices
            _sampling_indices = torch.randperm(buffer_size)

            # Start gradient steps loop
            for g in range(_num_gradient_step):
                # Get training batches
                batch_state_tensor = state_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]
                batch_action_tensor = action_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]
                batch_reward_tensor = reward_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]
                batch_next_state_tensor = next_state_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]

                # Prepare data for q learning
                y_true_1 = _getExpectedCumulativeReturn(inputs=batch_next_state_tensor)
                y_true_2 = _getExpectedQValues(inputs=torch.concat([batch_state_tensor, batch_action_tensor], dim=1))
                q_1_loss = _trainQ1Network(inputs=torch.concat([batch_state_tensor, batch_action_tensor], dim=1), y_true=y_true_1) # State-Action network (Q)
                q_2_loss = _trainVNetwork(inputs=batch_state_tensor, y_true=y_true_2)  # State-Value network (V)
                q_1_loss_accum += q_1_loss.tolist()
                q_2_loss_accum += q_2_loss.tolist()

                self.q_1_optimizer.zero_grad()
                self.q_2_optimizer.zero_grad()
                q_1_loss.backward(retain_graph=True)
                q_2_loss.backward()
                self.q_1_optimizer.step()
                self.q_2_optimizer.step()

                # Update target networks
                _updateTargetNetworks()

            # Append loss for recording.
            self.epsilon_beh_loss_append.append(diffu_loss_accum/_num_gradient_step)
            self.q_1_loss_append.append(q_1_loss_accum/_num_gradient_step)
            self.q_2_loss_append.append(q_2_loss_accum/_num_gradient_step)

            # Increase training record after epoch finished.
            _training_record += 1
            self.training_record += 1

            # Print the status.
            print('Epoch: ', self.training_record,
                  ', Gradient_step: ', int(self.training_record*_num_gradient_step), 
                  ', Diffu_loss: ', round(diffu_loss_accum/_num_gradient_step, 4),
                  ', Q1_loss: ', round(q_1_loss_accum/_num_gradient_step, 4),
                  ', Q2_loss: ', round(q_2_loss_accum/_num_gradient_step, 4),
                  ', Time/Epoch: ', round(time()-start_time, 4))
               
            # Save the training_records
            self.recordSaving(path=self.training_record_path)
            # Save the 2M training records (same gradient steps as baseline (SAC)).
            if self.training_record == 512 and not ft:
                self.recordSaving(path=self.training_record_2m_path)

            # Check for the breaking for evaluation.
            if eval and _training_record % 10 == 0:
                self.epsilon_beh.eval()
                self.q_1.eval()
                self.q_2.eval()
                break
        # ========================== Training Loop End ==========================

    # Training Record Saving Method
    def recordSaving(self, path:str):
        torch.save({'training_record': self.training_record,
                    'norm_reward_training_append': self.norm_reward_training_append,
                    'q_1': self.q_1,
                    'q_2': self.q_2, 
                    'q_1_tar': self.q_1_tar,
                    'q_2_tar': self.q_2_tar,
                    'epsilon_beh_loss_append': self.epsilon_beh_loss_append, 
                    'q_1_loss_append': self.q_1_loss_append, 
                    'q_2_loss_append': self.q_2_loss_append}, path)