

---

# MNIST Handwritten Digit Recognition

This project trains a neural network model to recognize handwritten digits using the MNIST dataset. The model is built using TensorFlow and Keras, and it can predict the digit from an external image.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Predicting Custom Digits](#predicting-custom-digits)
- [Results](#results)
- [License](#license)

## Overview
This project implements a simple neural network model to classify handwritten digits from the MNIST dataset. The model is trained and tested on the dataset, and it can also predict digits from external images.

## Dataset
The project uses the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which consists of 70,000 images of handwritten digits (60,000 for training and 10,000 for testing). Each image is 28x28 pixels in grayscale.

## Installation
To get started, clone this repository and install the required Python libraries:

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

### Required Libraries
- TensorFlow
- NumPy
- Matplotlib
- Pillow

You can install these dependencies using the following command:

```bash
pip install tensorflow numpy matplotlib pillow
```

## Model Architecture
The model consists of the following layers:
- **Flatten Layer**: Converts the 28x28 input images into a flat vector of 784 values.
- **Dense Layer**: A fully connected layer with 128 neurons and ReLU activation.
- **Output Layer**: A dense layer with 10 neurons (for 10 digit classes) and softmax activation for classification.

## Training
The model is trained for 10 epochs using the Adam optimizer and sparse categorical cross-entropy loss function. The training data is normalized to improve convergence.

```python
model.fit(x_train, y_train, epochs=10)
```

## Evaluation
After training, the model is evaluated on the test dataset:

```python
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

## Predicting Custom Digits
You can test the model's prediction on a custom image. Ensure that your image is a grayscale image with a resolution of 28x28 pixels.

1. Save your custom image as `digit.png`.
2. Use the following code to predict the digit:

```python
new_image = preprocess_image('digit.png')
prediction = model.predict(new_image)
predicted_digit = np.argmax(prediction)
print(f"Predicted Digit: {predicted_digit}")
```

### Preprocessing
The custom image is preprocessed to match the MNIST format by converting it to grayscale, inverting the colors (black to white), resizing to 28x28 pixels, and normalizing the pixel values.

## Results
After training, the model typically achieves an accuracy of around 98% on the test set.

Example output:
```
Accuracy: 98.15%
Predicted Digit: 7
```

