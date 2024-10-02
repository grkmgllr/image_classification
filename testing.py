import torch

from utils.model import NeuralNet
from utils.viz_utils import visualize_predictions

def test_model(device, test_loader):
    '''
    Tests a pre-trained neural network model on the CIFAR-10 test dataset and visualizes predictions.

    Args:
    - device: The device to use for testing (e.g., 'cuda' or 'cpu').
    - test_loader: DataLoader for the test dataset.

    Functionality:
    - Loads a pre-trained model from the file 'model/cifar10_model.pt'.
    - Evaluates the model on the test data.
    - Visualizes predictions using the `visualize_predictions` function for some of the test images.
    - Computes and prints the accuracy of the model on the entire test dataset.
    
    The function ensures no gradients are computed during the testing phase for efficiency and memory optimization.
    '''
    
    input_size = 32*32*3  # CIFAR-10 image size (3 channels, 32x32 pixels)
    num_classes = 10  # Number of classes in CIFAR-10
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 class labels

    # Initiliaze the model and then load the pre-trained model using torch.load
    model = NeuralNet(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load('model/cifar10_model.pt'))
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient computation during testing
    with torch.no_grad():
        correct = 0  # To count the number of correct predictions
        total = 0    # To count the total number of predictions
        figure_idx = 0  # For visualizing predictions every 2 batches

        # Loop over the test data
        for idx, (images, labels) in enumerate(test_loader):
            images = images.view(images.size(0), -1).to(device)  # Reshape the images and move to the device
            labels = labels.to(device)  # Move labels to the device

            # Forward pass: Compute predicted labels
            outputs      =  model(images) # Forward pass
            _, predicted =  torch.max(outputs, 1) # Get class with the highest probability

            total += labels.size(0)  # Increment the total count
            correct += (predicted == labels).sum().item()  # Increment the correct predictions count

            # Visualize predictions every 2 batches
            if idx % 2 == 0:
                visualize_predictions(images=images, predicted=predicted, actual=labels, idx=figure_idx, num_images=5, classes=classes)
                figure_idx += 1  # Increment the figure index for the next visualization
                
        # Print the accuracy of the model on the entire test dataset
        print(f'Accuracy of the network on the test images: {100 * correct / total:.2f} %')
