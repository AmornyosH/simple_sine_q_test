import torch

class MLP_Diffusion(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_Diffusion, self).__init__()
        num_nodes = 512
        self.fc1 = torch.nn.Linear(input_dim, num_nodes)
        self.fc2 = torch.nn.Linear(num_nodes, num_nodes)
        self.fc3 = torch.nn.Linear(num_nodes, num_nodes)
        self.output = torch.nn.Linear(num_nodes, output_dim)

    def forward(self, input):
        act1 = torch.nn.functional.mish(self.fc1(input))
        act2 = torch.nn.functional.mish(self.fc2(act1))
        act3 = torch.nn.functional.mish(self.fc3(act2))
        output = self.output(act3)
        return output

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, act_fn='mish'):
        super(MLP, self).__init__()
        num_nodes = 256
        self.fc1 = torch.nn.Linear(input_dim, num_nodes)
        self.fc2 = torch.nn.Linear(num_nodes, num_nodes)
        self.fc3 = torch.nn.Linear(num_nodes, num_nodes)
        self.output = torch.nn.Linear(num_nodes, output_dim)
        self.act_fn = act_fn

    def forward(self, input):
        if self.act_fn == 'mish':
            act1 = torch.nn.functional.mish(self.fc1(input))
            act2 = torch.nn.functional.mish(self.fc2(act1))
            act3 = torch.nn.functional.mish(self.fc3(act2))
        elif self.act_fn == 'relu':
            act1 = torch.nn.functional.relu(self.fc1(input))
            act2 = torch.nn.functional.relu(self.fc2(act1))
            act3 = torch.nn.functional.relu(self.fc3(act2))
        output = self.output(act3)
        return output

class MLP_GP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, act_fn='relu'):
        super(MLP_GP, self).__init__()
        num_nodes = 256
        self.fc1 = torch.nn.Linear(input_dim, num_nodes)
        self.fc2 = torch.nn.Linear(num_nodes, num_nodes)
        self.fc3 = torch.nn.Linear(num_nodes, num_nodes)
        self.output = torch.nn.Linear(num_nodes, output_dim)
        self.act_fn = act_fn

    def forward(self, input):
        if self.act_fn == 'mish':
            act1 = torch.nn.functional.mish(self.fc1(input))
            act2 = torch.nn.functional.mish(self.fc2(act1))
            act3 = torch.nn.functional.mish(self.fc3(act2))
        elif self.act_fn == 'relu':
            act1 = torch.nn.functional.relu(self.fc1(input))
            act2 = torch.nn.functional.relu(self.fc2(act1))
            act3 = torch.nn.functional.relu(self.fc3(act2))
        output = self.output(torch.nn.functional.tanh(act3))
        return output

class MLP_Qnetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_Qnetwork, self).__init__()
        num_nodes = 256
        self.fc1 = torch.nn.Linear(input_dim, num_nodes)
        self.fc2 = torch.nn.Linear(num_nodes, num_nodes)
        # self.fc3 = torch.nn.Linear(num_nodes, num_nodes)
        self.output = torch.nn.Linear(num_nodes, output_dim)

    def forward(self, input):
        act1 = torch.nn.functional.relu(self.fc1(input))
        act2 = torch.nn.functional.relu(self.fc2(act1))
        # act3 = torch.nn.functional.relu(self.fc3(act2))
        output = self.output(act2)
        return output

class CNN_1D(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN_1D, self).__init__()
        num_nodes = 256
        kernel_size = 3
        self.conv1 = torch.nn.Conv1d(input_dim, num_nodes, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = torch.nn.Conv1d(num_nodes, num_nodes, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3 = torch.nn.Conv1d(num_nodes, num_nodes, kernel_size=kernel_size, padding=kernel_size//2)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.output = torch.nn.Linear(num_nodes, output_dim)

    def forward(self, input):
        act1 = torch.nn.functional.mish(self.conv1(input))
        act2 = torch.nn.functional.mish(self.conv2(act1))
        act3 = torch.nn.functional.mish(self.conv3(act2))
        pool = self.global_pool(act3).squeeze(-1)
        output = self.output(pool)
        return output

class LNResNet(torch.nn.Module):
    def __init__(self,
                 input_dim:int, output_dim:int,
                 dropout_rate:float=0.00,
                 layer_norm_use:bool=False,
                 num_nodes:int=256,
                 num_resnet:int=3):
        super(LNResNet, self).__init__()
        self.num_nodes = num_nodes
        self.dropout_rate = dropout_rate
        self.layer_norm_use = layer_norm_use
        self.num_resnet = num_resnet

        # Input Dense layer
        self.fc1 = torch.nn.Linear(input_dim, num_nodes)
        # Add MLPResNet Block Components
        self.fc2 = torch.nn.Linear(self.num_nodes, self.num_nodes*4)
        self.fc3 = torch.nn.Linear(self.num_nodes*4, self.num_nodes)
        # Output Dense layer
        self.fc4 = torch.nn.Linear(self.num_nodes, output_dim)

    # MLPResNet Block
    def MLPResNet(self, inputs):
        _data = inputs
        if self.dropout_rate > 0:
            _data = torch.nn.Dropout(p=self.dropout_rate)(_data)
        if self.layer_norm_use:
            _data = torch.nn.LayerNorm()(_data)
        _data = self.fc2(_data)
        _data = torch.nn.functional.relu(_data)
        _data = self.fc3(_data)
        return _data + inputs

    def forward(self, inputs):
        _data = inputs
        _data = self.fc1(_data)
        for _ in range(self.num_resnet):
            _data = self.MLPResNet(_data)
        _data = torch.nn.functional.relu(_data)
        _data = self.fc4(_data)
        return _data

