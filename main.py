import torch
import torch.nn as nn
from utils.model import NeuralNet
from utils.data_utils import get_train_and_test_set, split_testset
from utils.viz_utils import visualize_train_val_losses
from training import train_model
from testing import test_model

def main():
    '''
    The main function that coordinates the training and testing of the neural network on the CIFAR-10 dataset.

    Steps:
    - Sets the input size and number of classes for the CIFAR-10 dataset.
    - Defines the training parameters such as epochs, batch size, and learning rate.
    - Initializes the model, loss function, and optimizer.
    - Loads the training, validation, and test datasets.
    - Trains the model and visualizes the training and validation losses.
    - Tests the trained model on the test dataset and prints the accuracy.

    '''

    input_size = 3*32*32  # CIFAR-10 images are 3-channel RGB images with 32x32 pixels (3*32*32)
    num_classes = 10  # CIFAR-10 has 10 classes
    epochs = 20 # Number of epochs for training
    batch_size = 64 # Number of samples per batch
    learning_rate = 0.001 # Learning rate for the optimizer

    # Define the device for computation (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, loss function, and optimizer
    model = NeuralNet(input_size=input_size, num_classes=num_classes).to(device) # Initiliaze and move the model to the device
    criterion = nn.CrossEntropyLoss() # Define the loss function (CrossEntropy for classification)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Choose optimizer with a learning rate

    # Get the training and test datasets
    train_set, test_set = get_train_and_test_set()
    
    # Create DataLoader for the training set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # Split the test set into validation and test subsets
    val_indices, test_indices = split_testset(test_set)
    
    # Create DataLoaders for validation and test subsets
    val_loader  = torch.utils.data.DataLoader(test_set, batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_indices))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_indices))
    
    # Train the model and track losses for each epoch
    train_losses, val_losses = train_model(model, device, epochs, criterion, optimizer, train_loader, val_loader)
    
    # Visualize the training and validation loss curves
    visualize_train_val_losses(train_losses, val_losses)

    # Test the model on the test dataset, get visualizations and print accuracy
    test_model(device, test_loader)

if __name__ == '__main__':
    main()
