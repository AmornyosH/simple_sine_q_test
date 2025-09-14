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
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        num_nodes = 256
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