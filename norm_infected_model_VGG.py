import torch 
import torch.nn as nn
import torch.nn.functional as F

# normal:0, infected: 1

# Define classifier class
class norm_infected_model_VGG(nn.Module):
    def __init__(self):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding = 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding = 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv3_1 = nn.Conv2d(128, 256, 3,padding = 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3,padding = 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3,padding = 1)
        self.conv3_4 = nn.Conv2d(256, 256, 4,padding = 1)
        self.fc1 = nn.Linear(18*18*256, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 1)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        x = F.relu(self.conv1_1(x))
        x = self.pool(F.relu(self.conv1_2(x)))
        x = F.relu(self.conv2_1(x))
        x = self.pool(F.relu(self.conv2_2(x)))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(F.relu(self.conv3_4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x) #F.sigmoid(x) is deprecated, use torch.sigmoid(x) instead