# Plant Leaf Disease Detection 🌿
This repository contains a machine learning project for detecting plant leaf diseases based on images. The aim is to assist farmers and agricultural professionals in identifying diseases early, improving crop health and productivity.

## Features
* Image-based disease detection for various plant leaves.
* Preprocessing techniques to enhance image quality for better model accuracy.
* Use of Convolutional Neural Networks (CNNs) for image classification.
* Evaluation metrics to assess model performance.


## Technologies Used
* Programming Language: Python
* Libraries:
  * TensorFlow/Keras for deep learning.
  * OpenCV and PIL for image preprocessing.
  * Matplotlib for visualizing results.
  * NumPy and Pandas for data manipulation.


## Dataset
The project uses a dataset of plant leaf images labeled by disease type ("healthy", "powdery", "rust").

Dataset source: https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset


## How It Works
1. Image Preprocessing:
   * Resize images to a consistent shape.
   * Normalize pixel values for faster training.
   * Apply data augmentation to improve generalization.
2. Model Training:
   * Train a CNN model on the dataset to classify leaf images by disease.
   * Use techniques like dropout and batch normalization for better accuracy.
3. Prediction:
   * The trained model predicts the disease based on an input image of a plant leaf.
4. Evaluation:
   * Model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.


## Installation
1. Clone this repository:
```
git clone https://github.com/lyviavalentina/Plant-Leaf-Disease-Detection.git
cd Plant-Leaf-Disease-Detection
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the script or notebook to train the model or make predictions.


## Future Enhancements
* Expand the dataset to include more plant species and diseases.
* Deploy the model as a web or mobile application for real-time disease detection.
* Integrate suggestions for disease treatment based on predictions.
