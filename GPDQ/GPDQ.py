'''
Gaussian Process Diffusion Q-learning
Version: v0
Revision: 1
Remark: PyTorch version
'''
# ============================ Pytorch Related ============================
import torch
CUDA = torch.cuda.is_available()
if CUDA:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
# ============================ Pytorch Related ============================

# ================================== Others ==================================
from GPDQ import my_exact_gp, my_svgp
from my_utilities import my_utils, my_NN
from time import time
import numpy as np
import os

class GaussianProcessDiffusionQlearning:
    def __init__(self, params_dict:dict, dataset:dict, ft:bool=False):
        self.ALG = 'GPDQ'
        self.ENV_CONFIG = params_dict['environment']
        self.STATE_DIM = int(params_dict['state_dim'])
        self.ACTION_DIM = int(params_dict['action_dim'])
        self.cuda = CUDA
        
        if dataset is not None:
            self.NUM_SAMPLE = dataset['observations'].shape[0]
            # Initialise replay buffers
            self.state_buffer = torch.tensor(dataset['observations'], dtype=torch.float32)        # State buffer (unnormalised)
            self.next_state_buffer = torch.tensor(dataset['next_observations'], dtype=torch.float32)   # Next state buffer (unnormalised)
            self.action_buffer = torch.tensor(dataset['actions'], dtype=torch.float32)       # Action buffer
            # Reward buffer
            if params_dict['normalise_reward']:
                self.reward_buffer = torch.tensor(self.normaliseRewards(dataset=dataset), dtype=torch.float32).view(-1, 1)
                print('Normalised_rewards!')
            else:
                self.reward_buffer = torch.tensor(dataset['rewards'], dtype=torch.float32).view(-1, 1)  
                print('Raw rewards!')
            self.MINIBATCH_SIZE = 256
        else:
            self.NUM_SAMPLE = 1e+06
            self.MINIBATCH_SIZE = 256

        # Initialise Diffusion Model's Parameters
        self.initialiseDiffusionParams(schedule='vp', beta_min=1, beta_max=10, num_step=params_dict['diffusion_step'], dec_step=params_dict['diffusion_step'])

        # Initialise Paths
        self.training_record_path = '{:s}/training_records/{:s}_{:s}_training_record'.format(self.ALG, self.ALG, self.ENV_CONFIG)
        self.evaluation_path = '{:s}/norm_eval_rewards_append'.format(self.ALG)
        self.q_eval_path = '{:s}/pred_q'.format(self.ALG)

        # Initialise neural networks
        self.EPSILON_INPUT_DIM = self.STATE_DIM + self.ACTION_DIM + self.POS_DIM
        self.EPSILON_BEH_INPUT_DIM = self.ACTION_DIM + self.POS_DIM
        self.Q_INPUT_DIM = self.STATE_DIM + self.ACTION_DIM
        self.V_INPUT_DIM = self.STATE_DIM
        # Check for the training record file and the response from user...
        if os.path.isfile(self.training_record_path) is True:
            print('========== ({:s}) There exists a training record for this agent. Do you wish to load the exist one ?'.format(self.ALG))
            _ans_1 = input('========== ({:s}) Press [y/n] and enter: '.format(self.ALG))
        else:
            _ans_1 = 'n'
        # Check for the answer...
        # ==================== Create new record and models ======================
        if _ans_1 == 'n' or _ans_1=='N' or _ans_1=='No' or _ans_1=='NO':
            print('========== ({:s}) Create new record and models!'.format(self.ALG))
            self.training_record = 0
            self.best_norm_reward_training = 0
            self.norm_reward_training_append = []
            
            self.q_1_loss_append = []
            self.q_2_loss_append = []

            self.q_1 = my_NN.MLP(input_dim=self.Q_INPUT_DIM, output_dim=1)
            self.q_2 = my_NN.MLP(input_dim=self.V_INPUT_DIM, output_dim=1)
            # Initialise Target models
            self.q_1_tar = self.q_1
            self.q_2_tar = self.q_2
            
            # Load pre-trained diffusion models
            self.beh_training_record = 0
            self.epsilon_beh = my_NN.MLP(input_dim=self.EPSILON_INPUT_DIM, output_dim=self.ACTION_DIM)
            self.epsilon_beh_loss_append = []

            # _loaded_training_record = torch.load(self.training_record_path, map_location=torch.device('cpu' if not CUDA else 'cuda'), weights_only=False)
            # self.beh_training_record = _loaded_training_record['beh_training_record']
            # self.epsilon_beh = _loaded_training_record['epsilon_beh']
            # self.epsilon_beh_loss_append = _loaded_training_record['epsilon_beh_loss_append']

        # ==================== Load Previous Record and Models ======================
        elif _ans_1 == 'y' or _ans_1=='Y' or _ans_1=='Yes' or _ans_1=='YES':
            print('========== ({:s}) Load training record.'.format(self.ALG))
            self.loadTrainingRecord()  # Load from the method here <--------- ****
            self.epsilon_beh.eval()
            self.q_1.eval()
            self.q_2.eval()
        else:
            print('========== ({:s}) Try another answer. ("y" for yes (load existing one) or "n" for no (create new)).'.format(self.ALG))
            exit()
            
        # Declare optimizer for the networks (offline training)
        self.epsilon_optimizer = torch.optim.Adam(self.epsilon_beh.parameters(), lr=3e-04)
        self.q_1_optimizer = torch.optim.Adam(self.q_1.parameters(), lr=3e-04)
        self.q_2_optimizer = torch.optim.Adam(self.q_2.parameters(), lr=3e-04)
        # Set to cuda if GPU is available.
        if CUDA:
            self.epsilon_beh.cuda()
            self.q_1.cuda()
            self.q_2.cuda()

        # ======================================= Create GP model. =======================================
        # Uncomment out the selected gp model type...
        self.gp_model_type = 'exact'
        # self.gp_model_type = 'sparse'
        # best_dataset = self.bestTrajExtraction(dataset=dataset) if dataset is not None else None
        # best_dataset = self.multiBestTrajExtraction(dataset=dataset, max_episode_steps=1000, top_k=10)
        best_dataset = dataset
        
        if self.gp_model_type == 'exact':
            self.gp_model = my_exact_gp.myExactGP(params_dict=params_dict, dataset=best_dataset, cuda=CUDA, parent_alg=self.ALG)
        else:
            print('No gp model type. Please try "exact" or "sparse".')
        
        # self.gp_model.y_train = self.getAlteredObservation(self.gp_model.x_train)

    # Training Record Loading Method
    def loadTrainingRecord(self, path:str=None):
        if path is None:
            _loaded_training_record = torch.load(self.training_record_path, map_location=torch.device('cpu' if not CUDA else 'cuda'), weights_only=False)
            # _loaded_training_record = torch.load(self.training_checkpoint_path, map_location=torch.device('cpu' if not CUDA else 'cuda'))
        else:
            _loaded_training_record = torch.load(path, map_location=torch.device('cpu' if not CUDA else 'cuda'))
        self.training_record = _loaded_training_record['training_record']
        self.beh_training_record = _loaded_training_record['beh_training_record']
        self.norm_reward_training_append = _loaded_training_record['norm_return_training_append']
        self.best_norm_reward_training = _loaded_training_record['best_norm_return_training']
        self.epsilon_beh = _loaded_training_record['epsilon_beh']
        self.q_1 = _loaded_training_record['q_1']
        self.q_2 = _loaded_training_record['q_2']
        self.q_1_tar = _loaded_training_record['q_1_tar']
        self.q_2_tar = _loaded_training_record['q_2_tar']
        self.epsilon_beh_loss_append = _loaded_training_record['epsilon_beh_loss_append']
        self.q_1_loss_append = _loaded_training_record['q_1_loss_append']
        self.q_2_loss_append = _loaded_training_record['q_2_loss_append']
        print('========== ({:s}) Training Record: '.format(self.ALG), self.training_record, ' epoch.', 
              ', Gradient steps: ', self.training_record*(self.NUM_SAMPLE//self.MINIBATCH_SIZE))

    # Diffusion model's parameters initialisation Method
    def initialiseDiffusionParams(self, schedule='vp', beta_min=0.1, beta_max=10, num_step=50, dec_step=10):
        # Intialise Diffusion Model Parameters
        self.DIFFU_STEPS = num_step
        self.DEC_DIFFU_STEPS = dec_step
        self.BETA_MIN = beta_min
        self.BETA_MAX = beta_max
        self.MIN_DIFFU_SPACE = torch.tensor(-1., dtype=torch.float32)
        self.MAX_DIFFU_SPACE = torch.tensor(1., dtype=torch.float32)
        self.POS_DIM = 4
        self.beta = np.zeros([self.DIFFU_STEPS, 1], dtype=float)
        self.alpha = np.zeros([self.DIFFU_STEPS, 1], dtype=float)
        self.alpha_bar = np.zeros([self.DIFFU_STEPS, 1], dtype=float)
        self.DIFFU_MEAN = torch.tensor(0., dtype=torch.float32)
        self.DIFFU_VAR = torch.tensor(1., dtype=torch.float32)
        self.DIFFU_STD = torch.sqrt(self.DIFFU_VAR)
        self.reverse_mean = 0.
        self.reverse_cov = 0.

        self.POS_EMB = my_utils.sinPositionEncoding(seq_len=self.DIFFU_STEPS, dim=self.POS_DIM)

        # ----- Derive the diffusion schedule
        # For the sake of simplicity, the order of the element in the array represents the forward process order.
        # For example, B_{fp} = [1, 2, 3, ..., N-1, N]  (the elements represent indices.)
        # In order to use for the reverse process, The elements have to be flip the other way around.
        # For example, B_{rp} = [N, N-1, N-2, ..., 2, 1] (the elements represent indices.)
        # Always remember that the diffusion process has the minimum time step is 1 and maximum at N. B = [1, 2, 3, ..., N]
        # ----- Variance Preserving Schedule
        if schedule == 'vp':
            # Derive the diffusion rate (noise schedule) 
            # which means, if it is the reverse, we have to reverse the order of the array.
            for i in range(self.DIFFU_STEPS):
                # Formular for the alpha: np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
                self.alpha[i] = np.exp((-self.BETA_MIN * 1/self.DIFFU_STEPS) - (0.5 * (self.BETA_MAX-self.BETA_MIN) * (2*(i+1)-1)/(np.square(self.DIFFU_STEPS))))
                self.beta[i] = 1 - self.alpha[i]
                self.alpha_bar[i] = np.prod(self.alpha[0:i+1])

        # ----- Linear Schedule (Same as original DDPM)
        elif schedule == 'linear':        
            self.beta = np.linspace(start=self.BETA_MIN, stop=self.BETA_MAX, num=self.DIFFU_STEPS)
            for i in range(self.DIFFU_STEPS):
                self.alpha[i] = 1 - self.beta[i]
                self.alpha_bar[i] = np.prod(self.alpha[0:i+1])
        
        # Convert them into tensor format.
        self.beta = torch.tensor(self.beta, dtype=torch.float32)
        self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        self.alpha_bar = torch.tensor(self.alpha_bar, dtype=torch.float32)
        self.POS_EMB = torch.tensor(self.POS_EMB, dtype=torch.float32)

    # Rewards Normalisation Method (Followed IQL) (Used for Gym-MuJoCo tasks only)
    def normaliseRewards(self, dataset):
        """
        IQL-style reward normalization with max 1000-step trajectories.
        Splits dataset, computes returns, normalizes rewards.
        """

        # === Step 1: Split dataset into trajectories (max 1000 steps each) ===
        observations = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards']
        terminals = dataset['terminals']
        next_observations = dataset['next_observations']

        trajectories = []
        current_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': []
        }
        step_count = 0  # To track steps within a trajectory (max 1000)

        for i in range(len(observations)):
            current_trajectory['observations'].append(observations[i])
            current_trajectory['actions'].append(actions[i])
            current_trajectory['rewards'].append(rewards[i])
            current_trajectory['next_observations'].append(next_observations[i])
            step_count += 1

            # Check for episode end: either terminal=True or max 1000 steps
            if terminals[i] or step_count >= 1000 or i == len(observations) - 1:
                trajectories.append({
                    'observations': np.array(current_trajectory['observations']),
                    'actions': np.array(current_trajectory['actions']),
                    'rewards': np.array(current_trajectory['rewards']),
                    'next_observations': np.array(current_trajectory['next_observations'])
                })
                # Reset for next trajectory
                current_trajectory = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'next_observations': []
                }
                step_count = 0

        print(f"Extracted {len(trajectories)} trajectories (max 1000 steps each).")

        # === Step 2: Compute returns ===
        returns = [np.sum(traj['rewards']) for traj in trajectories]
        returns_sorted = sorted(returns)
        R_min = returns_sorted[0]
        R_max = returns_sorted[-1]
        print(f"Return range: min={R_min:.2f}, max={R_max:.2f}")

        # === Step 3: Normalize rewards globally ===
        scaling_factor = 1.0 / (R_max - R_min + 1e-8)
        normalized_rewards = rewards * scaling_factor * 1000.0

        print(f"Rewards normalized with scaling factor: {scaling_factor:.6f}")

        return normalized_rewards

    # Best Trajectory Extraction Method (only 1)
    def bestTrajExtraction(self, dataset: dict, max_episode_steps=1000):
        observations = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards']
        terminals = dataset['terminals']
        next_observations = dataset['next_observations']

        num_samples = len(observations)

        trajectories = []
        current_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': []
        }
        step_count = 0  # To track steps within a trajectory (max 1000)

        for i in range(len(observations)):
            current_trajectory['observations'].append(observations[i])
            current_trajectory['actions'].append(actions[i])
            current_trajectory['rewards'].append(rewards[i])
            current_trajectory['next_observations'].append(next_observations[i])
            step_count += 1

            # Check for episode end: either terminal=True or max 1000 steps
            if terminals[i] or step_count >= max_episode_steps or i == len(observations) - 1:
                trajectories.append({
                    'observations': np.array(current_trajectory['observations']),
                    'actions': np.array(current_trajectory['actions']),
                    'rewards': np.array(current_trajectory['rewards']),
                    'next_observations': np.array(current_trajectory['next_observations'])
                })
                # Reset for next trajectory
                current_trajectory = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'next_observations': []
                }
                step_count = 0

        print(f"Extracted {len(trajectories)} trajectories (max 1000 steps each).")

        # Compute cumulative rewards
        cumulative_rewards = [np.sum(traj['rewards']) for traj in trajectories]

        # Find trajectory with highest cumulative reward
        max_idx = np.argmax(cumulative_rewards)
        best_traj = trajectories[max_idx]

        print(f"Highest cumulative reward: {cumulative_rewards[max_idx]:.2f}")

        # Prepare next_state
        obs = best_traj['observations']
        # next_obs = np.vstack([obs[1:], np.zeros_like(obs[0])])  # Last next_state = zeros (or could repeat last obs)
        next_obs = best_traj['next_observations']
        
        # Prepare final dataset dict
        best_dataset = {
            'arr_0': len(obs),  # Length of trajectory
            'arr_1': obs,
            'arr_2': best_traj['actions'],
            'arr_3': best_traj['rewards'],
            'arr_4': next_obs,
        }

        return best_dataset

    # Best Trajectory Extraction Method (>1)
    def multiBestTrajExtraction(self, dataset: dict, max_episode_steps=1000, top_k=10):
        observations = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards']
        terminals = dataset['terminals']
        next_observations = dataset['next_observations']

        num_samples = len(observations)

        trajectories = []
        current_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': []
        }
        step_count = 0  # Track steps in episode (max 1000)

        for i in range(num_samples):
            current_trajectory['observations'].append(observations[i])
            current_trajectory['actions'].append(actions[i])
            current_trajectory['rewards'].append(rewards[i])
            current_trajectory['next_observations'].append(next_observations[i])
            step_count += 1

            # End of trajectory
            if terminals[i] or step_count >= max_episode_steps or i == num_samples - 1:
                trajectories.append({
                    'observations': np.array(current_trajectory['observations']),
                    'actions': np.array(current_trajectory['actions']),
                    'rewards': np.array(current_trajectory['rewards']),
                    'next_observations': np.array(current_trajectory['next_observations'])
                })
                # Reset
                current_trajectory = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'next_observations': []
                }
                step_count = 0

        print(f"Extracted {len(trajectories)} trajectories (max 1000 steps each).")

        # Compute cumulative rewards
        cumulative_rewards = [np.sum(traj['rewards']) for traj in trajectories]

        # Get indices of top K cumulative rewards
        top_k_indices = np.argsort(cumulative_rewards)[-top_k:][::-1]  # descending order

        print(f"Top {top_k} cumulative rewards: {[cumulative_rewards[i] for i in top_k_indices]}")

        # Collect all top K trajectories and stack them
        all_obs = []
        all_actions = []
        all_rewards = []
        all_next_obs = []
        total_length = 0

        for idx in top_k_indices:
            traj = trajectories[idx]
            all_obs.append(traj['observations'])
            all_actions.append(traj['actions'])
            all_rewards.append(traj['rewards'])
            all_next_obs.append(traj['next_observations'])
            total_length += len(traj['observations'])

        # Stack everything
        stacked_obs = np.vstack(all_obs)
        stacked_actions = np.vstack(all_actions)
        stacked_rewards = np.hstack(all_rewards)  # rewards are usually 1D
        stacked_next_obs = np.vstack(all_next_obs)

        # Final dictionary
        best_dataset = {
            'arr_0': total_length,  # Total length of stacked trajectories
            'observations': stacked_obs,
            'actions': stacked_actions,
            'rewards': stacked_rewards.reshape(-1, 1),  # Shape [N, 1]
            'next_observations': stacked_next_obs,
        }

        return best_dataset

    # Diffusion process (forward process) Method
    def forwardProcess(self, data, epsilon, step:int=None):
        alpha_bars = self.alpha_bar[step]
        x_t_p_1 = (torch.sqrt(alpha_bars) * data) + (torch.sqrt(1-alpha_bars) * epsilon)
        return x_t_p_1

    # Original Reverse Process Method
    def reverseProcess(self, inputs:list, size:int, guide:bool=False, dec_step:bool=False):
        # Initialise noisy data (needed to be clipped).
        x_T = torch.normal(size=[size, self.ACTION_DIM], mean=self.DIFFU_MEAN, std=self.DIFFU_STD)
        _diffu_steps = self.DIFFU_STEPS if not dec_step else self.DEC_DIFFU_STEPS
        _mu_r = []
        _var_r = []
        # Predict guidance params
        if guide:
            _mu_r, _var_r = self.predictGP(x_test=inputs)
            self.mu_r = _mu_r
            self.mu_r = torch.clip(input=_mu_r, min=self.MIN_DIFFU_SPACE, max=self.MAX_DIFFU_SPACE)
            self.var_r = _var_r

        # Start reverse processes
        for i in range(_diffu_steps): 
            # Define reverse position indices 
            # (Have to be inversed since the diffusion schedule was created in forward process's order.)
            rev_pos = _diffu_steps-i-1
            rev_pos_emb = self.POS_EMB[rev_pos].repeat(size, 1)
            # Predict the noise for the reverse process (e_{theta})
            epsilon_theta_t = self.epsilon_beh(torch.concat((inputs, x_T, rev_pos_emb), dim=1))
            # Reverse process (Stochastic) (DDPM)
            x_t_m_1  = (x_T / torch.sqrt(self.alpha[rev_pos])) - \
                       (self.beta[rev_pos] * epsilon_theta_t / torch.sqrt(self.alpha[rev_pos]*(1-self.alpha_bar[rev_pos])))
            # Guided term
            if guide:
                _cov_dp = self.beta[rev_pos] * torch.eye(size) if i != (self.DIFFU_STEPS-1) else 0. * torch.eye(size)
                _cov_gp = torch.clip(self.var_r, min=self.beta[rev_pos]) * torch.eye(size, dtype=torch.float32)
                _inv_cov_gp = torch.linalg.inv(_cov_gp)
                x_t_m_1 += -torch.matmul(torch.matmul(_cov_dp, _inv_cov_gp), x_t_m_1 - self.mu_r)
                x_t_m_1 = torch.clip(x_t_m_1, min=self.MIN_DIFFU_SPACE, max=self.MAX_DIFFU_SPACE)

            if i == (_diffu_steps-1):
                x_t_m_1 += (torch.sqrt(self.beta[rev_pos]) * 0) 
            else:
                x_t_m_1 += (torch.sqrt(self.beta[rev_pos]) * torch.normal(size=[size, self.ACTION_DIM], mean=self.DIFFU_MEAN, std=self.DIFFU_STD))
            
            x_T = x_t_m_1  # Update the previous a (a_i) for the next iteration.

        return torch.clip(x_t_m_1, min=self.MIN_DIFFU_SPACE, max=self.MAX_DIFFU_SPACE)

    # Output prediction method
    def predict(self, state, size:int, guide:bool=True, dec_step:bool=False):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32).view(-1, self.STATE_DIM)
        return self.reverseProcess(state, size, guide, dec_step)

    # Altered observation computation Method 
    def getAlteredObservation(self, states):
        # print('Start computing altered memories...')
        with torch.no_grad():
            _y_hat_list = []
            for m in range(len(states)):
                # Altered Observation (My approach)
                # For q-network and diffusion, still using the original observation (x) not the latent.
                _sampling_size = 10
                _state_n_tensor = torch.reshape(states[m], [-1, self.STATE_DIM]).repeat(_sampling_size, 1)
                a_i_m_1 = self.predict(state=_state_n_tensor, size=_sampling_size, guide=False)

                # Get desired location (a, which max q)
                _q_values = self.q_1(torch.concat([_state_n_tensor, a_i_m_1], dim=1)) 

                _max_q_value = torch.max(_q_values)
                _max_q_indices = torch.where(_q_values == _max_q_value)[0]
                if _max_q_indices.shape[0] > 1:
                    _max_q_index = torch.reshape(_max_q_indices[0], [-1, 1])
                else: 
                    _max_q_index = _max_q_indices[0]
                _a_max = torch.squeeze(a_i_m_1[_max_q_index])
                _y_hat_list.append(_a_max)
        return torch.stack(_y_hat_list, dim=0).view(-1, self.ACTION_DIM)

    # Training Method (for offliine training)
    def training(self, total_epoch:int, eval:bool=False, ft:bool=False):
        # Get Expected Bellman's Equation (local)
        def _getExpectedCumulativeReturn(inputs):
            # return batch_reward_tensor + (_GAMMA * self.q_2(inputs))
            return batch_reward_tensor + (_GAMMA * self.q_1_tar(inputs))

        # Get Expected Q value Method (local)
        def _getExpectedQValues(inputs):
            return self.q_1_tar(inputs)

        # Q network Training Method (local)
        def _trainQ1Network(inputs, y_true):
            y_pred = self.q_1(inputs)
            _q_1_loss = torch.mean(torch.square(y_true - y_pred))
            return _q_1_loss

        # V network Training Method (local)
        def _trainVNetwork(inputs, y_true):
            # Compute gradients
            y_pred_tensor = self.q_2(inputs)
            _q_2_loss = torch.mean(torch.square(y_true-y_pred_tensor))
            # _q_2_loss = torch.sum(torch.square(y_true-y_pred_tensor) * torch.exp(_batch_dist.log_prob(_batch_pred_action)))
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

            # q_2_tar_state_dict = self.q_2_tar.state_dict()
            # q_2_state_dict = self.q_2.state_dict()
            # for key in q_2_state_dict:
            #     q_2_tar_state_dict[key] = q_2_state_dict[key]*_TAU + q_2_tar_state_dict[key]*(1-_TAU)
            # self.q_2_tar.load_state_dict(q_2_tar_state_dict)

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
        _gp_loss = 0
        _behaviour_record = 0

        # Set models to training mode.
        self.epsilon_beh.train()
        self.q_1.train()
        self.q_2.train()

        # ========================= Training Loop Start =========================
        # ========== Diffusion Models
        while self.beh_training_record < total_epoch:
            diffu_loss_accum = 0
            # Get shuffle indices
            _sampling_indices = torch.randperm(buffer_size)
            for g in range(_num_gradient_step):
                # Get training batches
                batch_state_tensor = state_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]
                batch_action_tensor = action_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]

                # Prepare data for diffusion learning
                rand_t = torch.randint(low=0, high=self.DIFFU_STEPS, size=[_batch_size])
                encode_t_tensor = self.POS_EMB[rand_t]  # Retrieve
                epsilon_tensor = torch.normal(mean=self.DIFFU_MEAN, std=self.DIFFU_STD, size=[_batch_size, self.ACTION_DIM])
                forward_action_tensor = self.forwardProcess(data=batch_action_tensor, epsilon=epsilon_tensor, step=rand_t)
                diffu_loss, _ = _trainDiffusionBeh(inputs=torch.concat([batch_state_tensor, forward_action_tensor, encode_t_tensor], dim=1), y_true=epsilon_tensor)
                diffu_loss_accum += diffu_loss.tolist()

            self.beh_training_record += 1
            self.epsilon_beh_loss_append.append(diffu_loss_accum/_num_gradient_step)
            print('Epoch: ', self.beh_training_record,
                    ', Gradient_step: ', int(self.beh_training_record*_num_gradient_step), 
                    ', Diffu_loss: ', round(diffu_loss_accum/_num_gradient_step, 4))
            # Save the training_records
            self.recordSaving(path=self.training_record_path)

        while self.training_record < total_epoch:
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
                _batch_next_action = self.predict(state=batch_next_state_tensor, size=_batch_size, guide=True, dec_step=False).view(-1, self.ACTION_DIM)
                y_true_1 = _getExpectedCumulativeReturn(inputs=torch.concat([batch_next_state_tensor, _batch_next_action], dim=1))

                # _batch_pred_action = self.predict(state=batch_state_tensor, size=_batch_size, guide=True, dec_step=False).view(-1, self.ACTION_DIM)
                # y_true_1 = _getExpectedCumulativeReturn(inputs=batch_next_state_tensor)
                
                # y_true_2 = _getExpectedQValues(inputs=torch.concat([batch_state_tensor, _batch_pred_action], dim=1))
                q_1_loss = _trainQ1Network(inputs=torch.concat([batch_state_tensor, batch_action_tensor], dim=1), y_true=y_true_1) # State-Action network (Q)
                q_1_loss_accum += q_1_loss.tolist()
                # q_2_loss = _trainVNetwork(inputs=batch_state_tensor, y_true=y_true_2) # State-Action network (Q)
                # q_2_loss_accum += q_2_loss.tolist()

                self.q_1_optimizer.zero_grad()
                # self.q_2_optimizer.zero_grad()
                q_1_loss.backward(retain_graph=True)
                # q_2_loss.backward(retain_graph=True)
                self.q_1_optimizer.step()
                # self.q_2_optimizer.step()s

                # # Prepare data for diffusion learning
                # rand_t = torch.randint(low=0, high=self.DIFFU_STEPS, size=[_batch_size])
                # encode_t_tensor = self.POS_EMB[rand_t]  # Retrieve
                # epsilon_tensor = torch.normal(mean=self.DIFFU_MEAN, std=self.DIFFU_STD, size=[_batch_size, self.ACTION_DIM])
                # forward_action_tensor = self.forwardProcess(data=batch_action_tensor, epsilon=epsilon_tensor, step=rand_t)
                # diffu_loss, _ = _trainDiffusionBeh(inputs=torch.concat([batch_state_tensor, forward_action_tensor, encode_t_tensor], dim=1), y_true=epsilon_tensor)
                # diffu_loss_accum += diffu_loss.tolist()

                # Update target networks
                _updateTargetNetworks()

            # Train the gp
            _gp_loss = self.gp_model.myTraining(total_epoch=10, ft=False)

            # Increase training record after epoch finished.
            self.training_record += 1

            # Append loss for recording.
            # self.epsilon_beh_loss_append.append(diffu_loss_accum/_num_gradient_step)
            self.q_1_loss_append.append(q_1_loss_accum/_num_gradient_step)
            self.q_2_loss_append.append(q_2_loss_accum/_num_gradient_step)

            # Print the status.
            print('Epoch: ', self.training_record,
                  ', Gradient_step: ', int(self.training_record*_num_gradient_step), 
                #   ', Diffu_loss: ', round(diffu_loss_accum/_num_gradient_step, 4),
                  ', Q1_loss: ', round(q_1_loss_accum/_num_gradient_step, 4),
                  ', GP_loss: ', round(_gp_loss, 4),
                  ', Time/Epoch: ', round(time()-start_time, 4))

            # Update Altered observation for every ... epoch.
            if self.training_record % 5 == 0:
                self.gp_model.y_train_full = self.getAlteredObservation(self.gp_model.x_train_full)
                # _gp_loss = self.gp_model.myTraining(total_epoch=10, ft=False)

            # Save the training_records
            self.recordSaving(path=self.training_record_path)

            # Check for the breaking for evaluation.
            if eval and self.training_record % 5 == 0:
                # self.epsilon_beh.eval()
                # self.q_1.eval()
                # self.q_2.eval()
                break

            # # if self.training_record % 5 == 0 and self.training_record > (total_epoch//16):
            # if self.training_record % 1 == 0 and self.training_record > (total_epoch//16):
            #     self.recordSaving(path=self.training_record_path+'_{:d}'.format(self.training_record))
            #     self.gp_model.recordSaving(path=self.gp_model.training_record_path+'_{:d}'.format(self.training_record))  # Save GP

        # ========================== Training Loop End ==========================

    # ==================================== GP model sections start ====================================
    # Prediction Method (mean, var)
    def predictGP(self, x_test):
        _mean, _var = self.gp_model.predict(X_s=x_test)
        return _mean, _var
    # ===================================== GP model sections end =====================================
    # Training Record Saving Method
    def recordSaving(self, path:str):
        torch.save({'training_record': self.training_record,
                    'beh_training_record': self.beh_training_record,
                    'norm_return_training_append': self.norm_reward_training_append,
                    'best_norm_return_training': self.best_norm_reward_training,
                    'epsilon_beh': self.epsilon_beh, 
                    'q_1': self.q_1,
                    'q_2': self.q_2, 
                    'q_1_tar': self.q_1_tar,
                    'q_2_tar': self.q_2_tar,
                    'epsilon_beh_loss_append': self.epsilon_beh_loss_append, 
                    'q_1_loss_append': self.q_1_loss_append, 
                    'q_2_loss_append': self.q_2_loss_append}, path)