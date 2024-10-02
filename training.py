import torch
import os

def train_model(model, device, epochs, loss_fn, optimizer, train_loader, val_loader):
    '''
    Trains a given neural network model and evaluates it on a validation set after each epoch.

    Args:
    - model (nn.Module): The neural network model to train.
    - device (torch.device): The device to use for training (e.g., 'cuda' or 'cpu').
    - epochs (int): Number of training epochs.
    - loss_fn (torch.nn.Module): The loss function to use (e.g., CrossEntropyLoss).
    - optimizer (torch.optim.Optimizer): The optimization algorithm (e.g., SGD, Adam).
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

    Returns:
    - train_losses (list of floats): The average training loss for each epoch.
    - val_losses (list of floats): The average validation loss for each epoch.
    
    The function also prints the training loss for each step and the validation loss and accuracy after each epoch.
    It saves the model at the end of each epoch to 'models/cifar10_model.pt'.
    '''
    train_losses = []  # List to store training loss for each epoch
    val_losses = []    # List to store validation loss for each epoch
    total_step = len(train_loader)  # Total number of training batches in each epoch

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_train_loss = 0.0  # Track training loss for the current epoch

        # Training Loop
        for i, (images, labels) in enumerate(train_loader):
            # Reshape images and move to device
            # Move labels to device
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            # Forward pass through the model
            outputs = model(images)
            
            # Compute the loss
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
              
            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()

            running_train_loss += loss.item()  # Accumulate loss

            # Print progress for each batch
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

        # Calculate average training loss for the current epoch
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Loop (evaluating model on validation set)
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient computation for validation
            for images, labels in val_loader:
                # Reshape images and move to device
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)
                
                # Move labels to device
                # Forward pass through the model
                # Compute validation loss
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, dim=1) # Get the class with the highest score using torch.max(outputs, dim=1)

                total += labels.size(0)               # Total number of labels
                correct += (predicted == labels).sum().item()  # Count correct predictions

        # Calculate average validation loss and accuracy for this epoch
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_accuracy = 100 * correct / total  # Validation accuracy
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Save the model at the end of each epoch as models/cifar10_model.pt
        if not os.path.exists('model'):
            os.makedirs('model')
        torch.save(model.state_dict(), 'model/cifar10_model.pt')
        
    return train_losses, val_losses # Return the list of training and validation losses
