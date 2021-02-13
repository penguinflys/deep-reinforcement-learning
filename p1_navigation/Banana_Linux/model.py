import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_dims = (64,64,64)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_dims(tuple(int)): dimensions of hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs = nn.ModuleList()
        last_dim = state_size
        for h_dim in hidden_dims:
            self.fcs.append(nn.Linear(last_dim,h_dim))
            last_dim = h_dim
        self.out = nn.Linear(last_dim,action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        output = state
        # print(output.device)
        for fc in self.fcs:
            output = F.relu(fc(output))
        
        return self.out(output)
