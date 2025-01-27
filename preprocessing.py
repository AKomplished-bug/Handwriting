import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def preprocess_image(image):
    """
    Preprocess a single image.
    - Resize to 32x32 pixels
    - Invert background color
    - Normalize pixel values
    """
    # Resize to 32x32 pixels
    image = cv2.resize(image, (32, 32))
    # Invert background color
    image = cv2.bitwise_not(image)
    # Normalize pixel values
    image = image / 255.0
    return image

def load_dataset(data_dir):
    """
    Load dataset from a directory structure.
    Directory format:
    data_dir/
        corrected/
        normal/
        reversal/
    """
    images = []
    labels = []
    label_map = {"corrected": 0, "normal": 1, "reversal": 2}  # Assign numeric labels to classes

    for label_name, label in label_map.items():
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' does not exist.")
            continue
        # Loop through all images in the class folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # Read and preprocess the image
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Unable to read file '{file_path}'. Skipping.")
                continue
            image = preprocess_image(image)
            images.append(image)
            labels.append(label)
    
    # Convert to numpy arrays
    images = np.array(images).reshape(-1, 32, 32, 1)  # Add channel dimension
    labels = np.array(labels)
    return images, labels

def load_and_preprocess_data(data_dir):
    """
    Load and preprocess the dataset and split it into training and testing sets.
    """
    images, labels = load_dataset(data_dir)
    labels = to_categorical(labels, num_classes=3)  # Convert labels to one-hot encoding
    return train_test_split(images, labels, test_size=0.3, random_state=42)

# Example usage
data_directory = "/mnt/c/Users/athul/Desktop/Handwriting/Data"  # Replace with the path to your dataset
X_train, X_test, y_train, y_test = load_and_preprocess_data(data_directory)

print(f"Training data: {X_train.shape}, Labels: {y_train.shape}")
print(f"Testing data: {X_test.shape}, Labels: {y_test.shape}")
