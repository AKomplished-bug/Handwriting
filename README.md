```markdown
# Dyslexia Detection System

This repository implements a system to detect dyslexia through handwriting analysis using a combination of a Convolutional Neural Network (CNN) and a Support Vector Machine (SVM). The project uses FastAPI to expose the functionality via an API.

---

## Features

- **CNN Model**: Pretrained CNN (`cnn_model.h5`) for extracting features from handwriting images.
- **SVM Model**: Pretrained SVM (`svm_classifier.pkl`) for classifying handwriting patterns into Normal, Reversal, or Corrected.
- **FastAPI Integration**: An API (`app.py`) for handling requests and processing handwriting images.
- **Inference Script**: A standalone script (`inference.py`) to classify images directly.
- **Preprocessing Module**: Handles preprocessing of input handwriting images (`preprocessing.py`).
- **Handwriting Page Splitting**: Logic implemented to split a full-page handwriting sample into meaningful segments for better classification.
- **Cumulative Assessment Logic**: A novel approach designed by our team to aggregate multiple classification results into a final dyslexia assessment.

---

## Project Structure

```
|-- app.py                 # FastAPI implementation for the API
|-- cnn_model.h5           # Pretrained CNN model for feature extraction
|-- inference.py           # Script for running classification locally
|-- model.py               # Script to train the CNN and SVM models
|-- preprocessing.py       # Utilities for preprocessing handwriting images
|-- requirements.txt       # Required Python dependencies
|-- svm_classifier.pkl     # Pretrained SVM model for classification
```

---

## Metadata Information

- **Dataset**: Handwriting samples used for training and testing, consisting of multiple categories (normal, reversal, corrected).
- **Classes**: Three primary classes - Normal, Reversal, Corrected.
- **Models Used**: CNN for feature extraction, SVM for classification.
- **Processing Pipeline**: Handwriting samples undergo preprocessing, segmentation, and classification.
- **API Endpoints**: FastAPI-based API to interact with the classification model.
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix analysis.

---

## Requirements

- Python 3.8 or higher
- TensorFlow
- OpenCV
- scikit-learn
- FastAPI
- Uvicorn

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Install Dependencies

Ensure that all required dependencies are installed:

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI Server

To start the FastAPI server, run:

```bash
uvicorn app:app --reload
```

The API will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## API Usage

### Endpoint: `/classify`

**Method**: POST

Uploads a handwriting image for classification.

#### Request
- **Content-Type**: multipart/form-data
- **Body**:
  - `file`: Handwriting image in grayscale (e.g., .jpg, .png)

#### Example:
Using curl:

```bash
curl -X POST "http://127.0.0.1:8000/classify" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path_to_image.jpg"
```

#### Response
- **200 OK**: JSON response with classification result:

```json
{
  "status": "success",
  "result": "Dyslexic"
}
```

---

## Running Inference Locally

Use the `inference.py` script to classify a handwriting image directly without running the API.

### Steps:
1. Ensure `cnn_model.h5` and `svm_classifier.pkl` are in the root directory.
2. Run the script:

```bash
python inference.py
```

3. Update the `page_image_path` variable in the script to point to your handwriting image.
4. The classification result will be printed in the console.

---

## Notes

- The handwriting image dataset is **not included in the repository** due to size constraints.
- Ensure images are in grayscale and properly preprocessed before feeding them into the model.
- Adjust padding or threshold parameters in `inference.py` if results are inconsistent for new datasets.

---

## Classification Report

### **Metrics Overview**
- **Precision**: Measures the proportion of true positive predictions out of all positive predictions for a class.
- **Recall**: Measures the proportion of true positive predictions out of all actual instances of a class.
- **F1-Score**: Harmonic mean of precision and recall.
- **Support**: Number of actual instances for each class in the dataset.

| Class  | Precision | Recall | F1-Score | Support  |
|--------|-----------|--------|----------|----------|
| Class 0 | 0.99 | 0.92 | 0.95 | 19,284 |
| Class 1 | 0.76 | 0.94 | 0.84 | 19,557 |
| Class 2 | 0.93 | 0.76 | 0.83 | 17,882 |

- **Accuracy**: 87%
- **Macro Average F1-Score**: 0.87
- **Weighted Average F1-Score**: 0.88

### **Confusion Matrix**

```
Confusion Matrix:
[[17753, 1461, 70],
 [213, 18319, 1025],
 [32, 4332, 13518]]
```

- **Diagonal Elements**: Correctly classified samples.
- **Off-Diagonal Elements**: Misclassified samples.

---

## References

This system was developed using methodologies inspired by various research papers. Key references include:
- [Research Paper on Handwriting Analysis for Dyslexia Detection] (add reference here)
- Our team implemented unique logic for **splitting a full-page handwriting sample** into meaningful segments.
- **Cumulative result logic** designed by us helps aggregate multiple assessment scores for an accurate dyslexia diagnosis.

---

## Troubleshooting

### Common Issues
1. **Model Loading Errors**:
   Ensure `cnn_model.h5` and `svm_classifier.pkl` are correctly trained and saved.
2. **Dependency Issues**:
   Check versions in `requirements.txt` and install missing libraries.
3. **API Not Running**:
   Ensure `uvicorn` is installed and accessible in your environment.

---

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

## Acknowledgments

- **TensorFlow**: For building and training the CNN model.
- **scikit-learn**: For SVM implementation.
- **FastAPI**: For easy and fast API creation.

For any questions or support, feel free to raise an issue in the repository!
```

