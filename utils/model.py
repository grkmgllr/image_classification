import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes, method='random_normal'):
        '''
        Initializes the neural network by setting up the layers and applying the selected weight initialization method.

        Args:
        - input_size (int): The number of input features.
        - num_classes (int): The number of output classes for classification.
        - method (str): The weight initialization method to be used for the layers. Options include:
            'xavier' for Xavier initialization,
            'kaiming' for Kaiming initialization,
            'random_normal' for random normal initialization.
        
        Layers:
        - fc1 to fc6: Fully connected layers with ReLU activation in between.
        - Consider this as a based architecture we have given you. You can play with it as you want.
        '''
        super(NeuralNet, self).__init__()

        # Define the layers
        self.fc1  = nn.Linear(input_size, 800) 
        self.fc2  = nn.Linear(800, 512)
        self.fc3  = nn.Linear(512, 456)
        self.fc4  = nn.Linear(456, 256)
        self.fc5  = nn.Linear(256, 100)
        self.fc6  = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()

        # Apply the chosen initialization method to the weights
        self.initialize_weights(method)
        
    def initialize_weights(self, method):
        '''
        Initializes the weights of the linear layers according to the specified method.

        Args:
        - method (str): The initialization method for the weights. Can be one of:
            - 'xavier': Applies Xavier uniform initialization.
            - 'kaiming': Applies Kaiming uniform initialization.
            - 'random_normal': Initializes weights from a normal distribution with mean 0 and standard deviation 0.01.

        '''
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]:
            if method == 'xavier':
                nn.init.xavier_uniform_(layer.weight)
            elif method == 'kaiming':
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            elif method == 'random_normal':
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.constant_(layer.bias, 0.0)


    def forward(self, x):
        '''
        Defines the forward pass of the neural network, passing the input through all the layers and ReLU activations.

        Args:
        - x (torch.Tensor): Input tensor containing the batch of data with dimensions matching the input_size.

        Returns:
        - torch.Tensor: The output tensor, typically used for classification.
        
        Layers are processed in the following order:
        - fc1 -> ReLU -> fc2 -> ReLU -> fc3 -> ReLU -> fc4 -> ReLU -> fc5 -> ReLU -> fc6.
        '''
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        out = self.fc6(x)

        return out
