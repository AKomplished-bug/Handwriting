from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tensorflow as tf
import joblib
import os

# Initialize FastAPI
app = FastAPI()

# Load the trained models
cnn_model = tf.keras.models.load_model('cnn_model.h5')
svm_model = joblib.load('svm_classifier.pkl')

# Define helper functions
def preprocess_image(image):
    """
    Preprocess a single letter image:
    - Resize to 32x32 pixels
    - Normalize pixel values
    """
    image = cv2.resize(image, (32, 32))
    image = cv2.bitwise_not(image)
    image = image / 255.0  
    return image

def extract_features(model, X):
    """
    Extract features using the second-last layer of the CNN model.
    """
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    features = feature_extractor.predict(X)
    return features

def classify_letter(cnn_model, svm_model, letter_image):
    """
    Classify a single letter using CNN and SVM.
    """
    letter_image = preprocess_image(letter_image)
    letter_image = letter_image.reshape(1, 32, 32, 1)  # Add batch and channel dimensions
    cnn_features = extract_features(cnn_model, letter_image)
    prediction = svm_model.predict(cnn_features)
    return prediction

def segment_and_classify_page(cnn_model, svm_model, page_image_path):
    """
    Process the handwriting page, segment letters, and classify them.
    """
    # Read the image in grayscale
    im = cv2.imread(page_image_path, 0)

    # Invert the colors of the image
    im = cv2.bitwise_not(im)

    # Apply binary thresholding
    _, thresh1 = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Define padding size
    padding = 10

    # Initialize counters for each class
    normal_count = 0
    reversal_count = 0
    corrected_count = 0

    # Process each contour
    for cnt in contours:
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out non-letter contours based on aspect ratio and size
        aspect_ratio = w / h
        area = cv2.contourArea(cnt)

        if 0.2 < aspect_ratio < 1.0 and w > 15 and h > 15 and area > 50:  # Adjust thresholds as needed
            # Add padding while ensuring dimensions stay within image bounds
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(thresh1.shape[1], x + w + padding) - x_pad
            h_pad = min(thresh1.shape[0], y + h + padding) - y_pad

            # Crop the letter with padding
            cropped = thresh1[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]

            # Classify the letter
            prediction = classify_letter(cnn_model, svm_model, cropped)

            # Update counters based on prediction
            if prediction == 0:
                normal_count += 1
            elif prediction == 1:
                reversal_count += 1
            elif prediction == 2:
                corrected_count += 1

    # Aggregate results
    total_letters = normal_count + reversal_count + corrected_count
    if total_letters == 0:
        return {"message": "No letters detected"}

    reversal_ratio = reversal_count / total_letters
    corrected_ratio = corrected_count / total_letters

    # Define threshold for dyslexia classification
    threshold = 0.3 

    if reversal_ratio + corrected_ratio > threshold:
        return {"classification": "Dyslexic", "reversal_ratio": reversal_ratio, "corrected_ratio": corrected_ratio}
    else:
        return {"classification": "Non-Dyslexic", "reversal_ratio": reversal_ratio, "corrected_ratio": corrected_ratio}

# Define FastAPI endpoint
@app.post("/classify-page/")
async def classify_page(file: UploadFile = File(...)):
    """
    Endpoint to classify a handwriting page as dyslexic or non-dyslexic.
    """
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.file.read())

        # Process the page and get results
        result = segment_and_classify_page(cnn_model, svm_model, temp_file_path)

        # Remove the temporary file
        os.remove(temp_file_path)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
