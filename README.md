# Brain Tumor Detection using Deep Learning

This project focuses on the detection and classification of brain tumors from MRI scans using deep learning techniques. Two powerful pre-trained convolutional neural network (CNN) architectures, **ResNet101** and **EfficientNetB0**, are leveraged for this task through transfer learning.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abhisheksinha31/Brain-Tumor-Detection/blob/main/Resnet_%26_EfficientNet_Brain_Mri.ipynb)

## Dataset

The project utilizes a dataset of brain MRI images, which are categorized into four classes:
1.  **Glioma Tumor**
2.  **Meningioma Tumor**
3.  **Pituitary Tumor**
4.  **No Tumor**

The images are sourced from training and testing folders and are preprocessed before being fed into the models.

## Methodology

The workflow for this project is as follows:

1.  **Data Preparation**: Images are loaded from the respective class folders. Each image is resized to `150x150` pixels.
2.  **Data Splitting**: The dataset is shuffled and then split into a training set and a testing set.
3.  **One-Hot Encoding**: The categorical labels are converted into a one-hot encoded format suitable for the models.
4.  **Model Building**: Two separate models are built using pre-trained architectures.
5.  **Training**: Both models are trained on the training data. Callbacks like `ReduceLROnPlateau` and `ModelCheckpoint` are used to optimize the training process.
6.  **Evaluation**: The performance of each model is evaluated on the test set using metrics like accuracy, confusion matrix, and a classification report.
7.  **Prediction**: A function is available to predict the class of a random image from the test set and display the result.

## Models

Two deep learning models are implemented using transfer learning:

### 1. ResNet101
A pre-trained ResNet101 model is used as the base. The top classification layer is removed and replaced with a custom head consisting of:
- `GlobalAveragePooling2D`
- `Dropout` (with a rate of 0.5)
- `Dense` layer with a `softmax` activation function for 4-class classification.

### 2. EfficientNetB0
Similarly, a pre-trained EfficientNetB0 model is used. The custom classification head includes:
- `GlobalAveragePooling2D`
- `Dense` layer (1024 units, ReLU activation)
- `Dropout` (with a rate of 0.4)
- `Dense` layer with `softmax` activation for the final classification.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Abhisheksinha31/Brain-Tumor-Detection.git
    cd Brain-Tumor-Detection
    ```

2.  **Prerequisites**: Ensure you have the necessary libraries installed.
    ```bash
    pip install tensorflow numpy pandas seaborn matplotlib opencv-python mplcyberpunk
    ```

3.  **Dataset**: Download the dataset and place it in the appropriate directory structure as referenced in the notebook (e.g., `../content/drive/MyDrive/archive/`).

4.  **Run the Notebook**: Open and run the `Resnet_&_EfficientNet_Brain_Mri.ipynb` notebook in a Jupyter environment or Google Colab.
