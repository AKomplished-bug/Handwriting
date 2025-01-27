import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pickle
import preprocessing
# CNN Architecture for Feature Extraction
def build_cnn():
    """
    Build a CNN for feature extraction.
    Returns a Keras model.
    """
    input_layer = Input(shape=(32, 32, 1))  # Input size: 32x32 grayscale images
    
    # Convolutional Layer 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Convolutional Layer 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Convolutional Layer 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Fully Connected Layer
    x = Flatten()(x)
    feature_layer = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(feature_layer)
    
    # Output Layer for CNN only (Softmax classifier)
    output_layer = Dense(3, activation='softmax')(x)
    
    # Create the model
    cnn_model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model for training
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return cnn_model, Model(inputs=input_layer, outputs=feature_layer)  # Full CNN and feature extractor

# Train the CNN Model
def train_cnn(cnn_model, X_train, y_train, X_val, y_val, epochs=30, batch_size=64):
    """
    Train the CNN model.
    """
    history = cnn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history

# Train SVM on CNN Features
def train_svm(feature_extractor, X_train, y_train, X_test, y_test):
    """
    Train an SVM classifier on CNN-extracted features.
    """
    # Extract features using the CNN feature extractor
    X_train_features = feature_extractor.predict(X_train, batch_size=64)
    X_test_features = feature_extractor.predict(X_test, batch_size=64)
    
    # Flatten features for SVM input
    X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)
    X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)
    
    # Convert labels to numeric for SVM
    y_train_numeric = np.argmax(y_train, axis=1)
    y_test_numeric = np.argmax(y_test, axis=1)
    
    # Train the SVM
    svm_classifier = SVC(kernel='rbf', C=1, gamma='scale')
    svm_classifier.fit(X_train_features, y_train_numeric)
    
    # Evaluate the SVM
    y_pred = svm_classifier.predict(X_test_features)
    print("Classification Report:")
    print(classification_report(y_test_numeric, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_numeric, y_pred))
    
    return svm_classifier

# Main Training Workflow
def train_cnn_svm(X_train, X_val, y_train, y_val, X_test, y_test, epochs=30, batch_size=64):
    """
    Train the CNN-SVM hybrid model.
    """
    # Step 1: Build and train the CNN
    cnn_model, feature_extractor = build_cnn()
    print("Training CNN...")
    train_cnn(cnn_model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    
    # Step 2: Train the SVM using CNN features
    print("Training SVM on CNN features...")
    svm_classifier = train_svm(feature_extractor, X_train, y_train, X_test, y_test)
    
    # Save models for future use
    cnn_model.save("cnn_model.h5")
    with open("svm_classifier.pkl", "wb") as f:
        pickle.dump(svm_classifier, f)
    print("Models saved: cnn_model.h5 and svm_classifier.pkl")

# Example Usage
if __name__ == "__main__":
    # Load preprocessed data
    train_directory = "/mnt/c/Users/athul/Desktop/Handwriting/Data/train"
    test_directory = "/mnt/c/Users/athul/Desktop/Handwriting/Data/test"
    X_train, X_test, y_train, y_test = preprocessing.load_and_preprocess_train_test_data(train_directory, test_directory)
    
    # Split training data for validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Train the hybrid model
    train_cnn_svm(X_train, X_val, y_train, y_val, X_test, y_test, epochs=30, batch_size=64)
