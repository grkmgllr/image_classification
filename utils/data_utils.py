import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def transform_cifar10():
    '''
    Defines the transformations for the CIFAR-10 dataset.
    
    Returns:
    - transform (torchvision.transforms.Compose): A composition of transformations that:
      - Converts the image to a PyTorch tensor.
      - Normalizes the image with a mean and standard deviation of 0.5 for each RGB channel.
    
    This transformation is used for training, validation, and testing.
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return transform

def split_testset(test_set, test_size=0.5):
    '''
    Splits the test set into two subsets: validation and test.

    Args:
    - test_set (torchvision.datasets.CIFAR10): The full CIFAR-10 test dataset.
    - test_size (float): Proportion of the test set to allocate to the actual test set, with the remainder used for validation.

    Returns:
    - val_indices (list): Indices for the validation subset.
    - test_indices (list): Indices for the test subset.

    The function uses `train_test_split` from `sklearn` to split the indices of the test set.
    '''
    test_indices = list(range(len(test_set))) # List of all indices in the test set
    val_indices, test_indices = train_test_split(test_indices, test_size=test_size, random_state=42)

    return val_indices, test_indices

def get_train_and_test_set():
    '''
    Downloads and transforms the CIFAR-10 training and test datasets.

    Returns:
    - train_set (torchvision.datasets.CIFAR10): The CIFAR-10 training dataset after applying transformations.
    - test_set (torchvision.datasets.CIFAR10): The CIFAR-10 test dataset after applying transformations.

    The CIFAR-10 dataset is downloaded if it is not already available in the specified directory.
    '''
    transform = transform_cifar10() # Get the transformations using transform_cifar10 function
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # Get train set
    test_set  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) # Get test set

    return train_set, test_set

def get_trainloader(train_set, batch_size):
    '''
    Creates a DataLoader for the training dataset.

    Args:
    - train_set (torchvision.datasets.CIFAR10): The CIFAR-10 training dataset.
    - batch_size (int): The number of samples per batch to load.

    Returns:
    - trainloader (torch.utils.data.DataLoader): A DataLoader that provides an iterable over the training dataset.

    The DataLoader shuffles the data after each epoch to ensure better training performance.
    '''
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
    return trainloader

def get_testloader(test_set, test_indices, batch_size):
    '''
    Creates a DataLoader for the test dataset, using only a subset of the dataset.

    Args:
    - test_set (torchvision.datasets.CIFAR10): The full CIFAR-10 test dataset.
    - test_indices (list): Indices for the test subset.
    - batch_size (int): The number of samples per batch to load.

    Returns:
    - test_loader (torch.utils.data.DataLoader): A DataLoader that provides an iterable over the test subset.

    The function uses a subset of the test dataset based on the provided indices.
    It does not shuffle the data, as shuffling is unnecessary during testing.
    '''
    test_set = Subset(test_set, test_indices) # Create a subset based on test indices using Subset
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=3) 

    print(f'Number of test samples: {len(test_set)}')

    return test_loader

def get_validationloader(test_set, val_indices, batch_size):
    '''
    Creates a DataLoader for the validation dataset, using a subset of the original test dataset.

    Args:
    - test_set (torchvision.datasets.CIFAR10): The full CIFAR-10 test dataset.
    - val_indices (list): Indices for the validation subset.
    - batch_size (int): The number of samples per batch to load.

    Returns:
    - val_loader (torch.utils.data.DataLoader): A DataLoader that provides an iterable over the validation subset.

    The function uses a subset of the test dataset as the validation set, based on the provided indices.
    It does not shuffle the data during validation.
    '''
    valset =  Subset(test_set, val_indices) # Create a subset for validation using Subset
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=3)

    print(f'Number of validation samples: {len(valset)}')

    return val_loader
