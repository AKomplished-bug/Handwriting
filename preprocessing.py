import os
import cv2
import numpy as np
import cupy as cp  # Import CuPy for GPU acceleration
from tensorflow.keras.utils import to_categorical

def preprocess_image(image):
    """
    Preprocess a single image using CUDA (CuPy):
    - Resize to 32x32 pixels
    - Invert background color
    - Normalize pixel values
    """
    # Resize to 32x32 pixels using OpenCV
    image = cv2.resize(image, (32, 32))
    # Convert image to CuPy array
    image = cp.asarray(image, dtype=cp.uint8)  # Use uint8 for bitwise operations
    # Invert background color
    image = cp.bitwise_not(image)
    # Convert to float32 and normalize pixel values
    image = image.astype(cp.float32) / 255.0
    # Convert back to NumPy for compatibility
    return cp.asnumpy(image)


def load_dataset_from_directory(data_dir):
    """
    Load images and labels from a directory structure using CUDA (CuPy) for preprocessing.
    Directory format:
    data_dir/
        corrected/
        normal/
        reversal/
    """
    images = []
    labels = []
    label_map = {"Corrected": 0, "Normal": 1, "Reversal": 2}  # Map labels to numeric values

    for label_name, label in label_map.items():
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' does not exist. Skipping.")
            continue
        # Iterate through all image files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # Read the image in grayscale
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Unable to read file '{file_path}'. Skipping.")
                continue
            # Preprocess the image using CUDA
            image = preprocess_image(image)
            images.append(image)
            labels.append(label)
    
    # Convert lists to numpy arrays
    images = np.array(images).reshape(-1, 32, 32, 1)  # Add channel dimension
    labels = np.array(labels)
    return images, labels

def load_and_preprocess_train_test_data(train_dir, test_dir):
    """
    Load and preprocess training and testing datasets using CUDA for preprocessing.
    """
    # Load training data
    X_train, y_train = load_dataset_from_directory(train_dir)
    # Load testing data
    X_test, y_test = load_dataset_from_directory(test_dir)

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)

    return X_train, X_test, y_train, y_test

train_directory = "/mnt/c/Users/athul/Desktop/Handwriting/Data/train"  
test_directory = "/mnt/c/Users/athul/Desktop/Handwriting/Data/test"    

X_train, X_test, y_train, y_test = load_and_preprocess_train_test_data(train_directory, test_directory)

print(f"Training data: {X_train.shape}, Labels: {y_train.shape}")
print(f"Testing data: {X_test.shape}, Labels: {y_test.shape}")
