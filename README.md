# Classification-of-bronzes-by-CNN

This project is our final assignment for the course **Science, Technology, and Society in Prehistoric China**. We attempt to use Convolutional Neural Networks (CNN) to distinguish different types of bronze artifacts.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The study of bronze artifacts provides invaluable insights into the technological advancements and social structures of ancient China. Our project focuses on leveraging modern machine learning techniques to classify various types of bronze artifacts, thus aiding archaeologists and historians in their research.

## Dataset

The dataset used for this project includes high-resolution images of different types of bronze artifacts. These images were collected from various archaeological databases and publications. Each image is labeled according to the artifact type.

## Methodology

We utilized a Convolutional Neural Network (CNN) for the classification task. The CNN architecture was chosen due to its proven effectiveness in image recognition tasks. The model was trained using TensorFlow and Keras libraries.

Key steps in our methodology include:
- Data Preprocessing: Resizing images, normalization, and augmentation.
- Model Architecture: Designing and implementing the CNN model.
- Training: Training the model on the dataset with appropriate hyperparameters.
- Evaluation: Assessing the model's performance using metrics such as accuracy, precision, and recall.

## Contributing

We welcome contributions to improve this project. If you have any suggestions or bug reports, please open an issue or submit a pull request.

## Results

The RESNET50 model(from torch.nn.Module) achieved an accuracy of XX% on the test dataset. 

## License

MIT
