# Brain Tumor Classification using Convolutional Neural Networks (CNN)

## Overview

In this project, we develop a **machine learning model** using **TensorFlow** and the **Keras API** to classify **brain MRI scans** as either **healthy** or **unhealthy** (presence of a brain tumor). The model utilizes **Convolutional Neural Networks (CNN)** for image recognition in the field of medical diagnostics. The primary goal of this project is to provide a reliable and automated tool for classifying brain scans based on their condition.

## Problem Definition

Given a set of MRI brain scans, the objective is to classify each image as either:
- **Healthy Brain (No Tumor)**
- **Unhealthy Brain (Tumor Present)**

This is a **binary image classification** problem, where the model must learn to distinguish between the two categories: healthy and unhealthy brain scans.

## Dataset

The dataset used for this project is the **Brain Tumor Dataset** from Kaggle, which contains **4,600 MRI brain scans** divided into two folders:
- **Healthy**: MRI scans of healthy brains
- **Unhealthy**: MRI scans showing tumors or unhealthy brains

### Dataset Source:
- [Brain Tumor Dataset on Kaggle](https://www.kaggle.com/datasets)

## Approach

We approach the problem using **Convolutional Neural Networks (CNN)**, which are specifically designed for image classification tasks. CNNs are highly efficient in recognizing patterns and structures in images, making them ideal for tasks such as medical image classification.

### Steps Involved:
1. **Data Preprocessing**:
   - Organize images into the corresponding classes: `Healthy` and `Unhealthy`.
   - Resize images to a uniform size for feeding into the neural network.
   - Normalize pixel values to improve model performance.
   
2. **Model Architecture**:
   - **Input Layer**: The raw pixel values of the image (Resized to a fixed size).
   - **Convolutional Layers (Conv2D)**: These layers will learn local patterns (such as edges, textures, and more complex shapes).
   - **ReLU Activation Function**: Introduces non-linearity into the model and helps solve the vanishing gradient problem.
   - **Max Pooling Layers**: Reduce spatial dimensions, making the network more computationally efficient.
   - **Fully Connected Layer (Dense)**: This layer makes predictions by using the learned features from previous layers.
   - **Output Layer**: A single neuron with a sigmoid activation function to classify the image as either healthy (0) or unhealthy (1).
   
3. **Model Training**:
   - Use **binary cross-entropy** as the loss function since it's a binary classification problem.
   - Optimizer: **Adam** (adaptive moment estimation) for faster convergence.
   - Train the model on the dataset and evaluate using accuracy.

4. **Model Evaluation**:
   - Metrics: **Accuracy**, **Precision**, **Recall**, **F1 Score**.

5. **Deployment**:
   - Save the trained model using **TensorFlow** and deploy it as a **Flask web application** for real-time prediction.

## Model Architecture

The **CNN architecture** consists of the following layers:
- **Input Layer**: Takes the image as input, where each image is resized to a uniform size (e.g., 150x150 pixels).
- **Conv2D Layer**: Two convolutional layers with 32 and 64 filters, respectively, and a kernel size of (3, 3).
- **MaxPooling Layer**: Two max-pooling layers with a pool size of (2, 2) to reduce the dimensionality.
- **Fully Connected Dense Layer**: 512 neurons to process the extracted features.
- **Output Layer**: A single neuron with a sigmoid activation to classify images as either healthy or unhealthy.
  
### CNN Implementation Steps:
1. **Convolution Layer**: This layer computes the output of neurons that are connected to local regions in the input. Each neuron computes a dot product between its weights and a small region of the input image.
2. **ReLU Activation**: The rectified linear unit introduces nonlinearity and solves the vanishing gradient problem.
3. **MaxPooling Layer**: The pooling operation reduces the spatial dimensions of the image, improving the model's computational efficiency.
4. **Fully Connected Layer**: Connects all neurons from the previous layer to a single output neuron to make predictions.
  
### Implementation Tools:
- **TensorFlow**: A popular deep learning framework for building and training models.
- **Keras API**: High-level neural networks API for building the CNN.
- **Flask**: For deploying the trained model as a web application for real-time predictions.

## Model Evaluation

After training the model, it was evaluated using standard classification metrics:

- **Accuracy**: Measures the overall correctness of the model.
- **Precision**: Measures the accuracy of positive predictions (i.e., predicting tumors).
- **Recall**: Measures the ability of the model to correctly identify unhealthy scans.
- **F1 Score**: The harmonic mean of precision and recall, useful for imbalanced datasets.

## Results

The model's final evaluation results are:
- **Accuracy**: 95% (on validation data)
- **Precision**: 94%
- **Recall**: 96%
- **F1 Score**: 95%

The model performs very well, showing high accuracy and recall, which indicates it is good at identifying unhealthy scans (i.e., brain tumors).

## Deployment

Once the model is trained and evaluated, it is saved using **TensorFlow's model saving functionality**. The model can then be deployed as a real-time image classification web app using **Flask**. The web application allows users to upload an MRI image and receive a prediction on whether the image represents a healthy brain or a brain with a tumor.

### Steps to Deploy the Model:
1. Train the model and save it using `model.save('brain_tumor_model.h5')`.
2. Create a **Flask application** to load the trained model.
3. Provide an interface to upload MRI images and classify them using the trained model.

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/VishakhViswanath/brain_tumor_detection-using-CNNs.git

2. Install the required dependencies:

   pip install -r requirements.txt

3. Train the model by running the train_model.py script:

   python train_model.py

4. For deployment, run the Flask app:

    python app.py

    Open the browser and visit http://127.0.0.1:5000/ to interact with the web application.

5. Conclusion

This project demonstrates the power of Convolutional Neural Networks (CNNs) in medical image classification. By training on MRI scans, the model is able to   accurately   classify brain scans as either healthy or unhealthy (tumor present). With the potential for real-time deployment, this model can aid medical professionals in early detection and diagnosis of brain tumors, ultimately improving healthcare outcomes.

6. Future Work

    Improve model performance by experimenting with more complex CNN architectures or transfer learning.
    Incorporate additional features such as MRI scan metadata (e.g., patient age, gender) to enhance model predictions.
    Develop a more robust web application with advanced user interfaces and data visualization.
   

    Dataset provided by Kaggle Brain Tumor Dataset.
    Libraries used: TensorFlow, Keras, Flask, NumPy, OpenCV
