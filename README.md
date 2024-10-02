📦 CIFAR-10 Neural Network Classifier

This project implements a neural network to classify images from the CIFAR-10 dataset, consisting of 60,000 32x32 color images in 10 classes, using PyTorch. The project includes data loading, model training, evaluation, and visualization utilities.


📝 Project Overview

The goal of this project is to build and train a neural network for image classification on the CIFAR-10 dataset. Key features include:

- Data Augmentation and loading utilities.
- Training and Validation Loss Visualization.
- Model Testing with sample predictions.
- Accuracy Calculation on the test dataset.


🚀 Usage


🏋️‍♂️ Training
To train the neural network:

- Load the CIFAR-10 dataset.
- Train the neural network for 20 epochs (configurable).
- Visualize the training and validation loss curves.
- Save the trained model to the model/ directory as cifar10_model.pt.


🧪 Testing
After training, test the model's performance on the test dataset:

- Load the saved model.
- Test it on the CIFAR-10 test dataset.
- Output the test accuracy and visualize sample predictions.


🧠 Model Architecture

The model is a fully connected neural network with:

- Input Layer: Size of 3*32*32 (flattened CIFAR-10 image).
- Hidden Layers: ReLU activations.
- Output Layer: 10 units (one for each CIFAR-10 class).
- Loss Function: CrossEntropyLoss for classification.


📊 Results Visualization

Loss Curves:
After training, the loss curves for both training and validation are plotted.


📦 Dataset

The CIFAR-10 dataset contains the following classes:

✈️ Airplane
🚗 Automobile
🐦 Bird
🐱 Cat
🦌 Deer
🐕 Dog
🐸 Frog
🐎 Horse
🚢 Ship
🚚 Truck
The dataset is automatically downloaded when you run the project.

🤝 Contributing

Contributions are welcome! If you'd like to improve the project or have suggestions, feel free to open an issue or submit a pull request.

📝 License

This project is licensed under the MIT License. See the LICENSE file for more details.

Feel free to replace your-username in the clone command with your actual GitHub username, and customize the text further as needed!
