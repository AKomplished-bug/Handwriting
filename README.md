# Handwriting
This classification report and confusion matrix provide a detailed analysis of the performance of a multi-class classification model. Let’s break it down:

---

### **Classification Report**

1. **Classes**:
   - **Class 0**: Represents one category of the dataset.
   - **Class 1**: Represents another category.
   - **Class 2**: Represents the third category.

2. **Metrics**:
   - **Precision**: Measures the proportion of true positive predictions out of all positive predictions for a class. High precision means fewer false positives.
   - **Recall**: Measures the proportion of true positive predictions out of all actual instances of a class. High recall means fewer false negatives.
   - **F1-Score**: Harmonic mean of precision and recall. It balances the two metrics, especially useful when classes are imbalanced.
   - **Support**: The number of actual instances for each class in the dataset.

#### **Class 0**:
   - **Precision**: 0.99 (Excellent, very few false positives).
   - **Recall**: 0.92 (High recall, most true instances were identified).
   - **F1-Score**: 0.95 (Strong overall performance for this class).
   - **Support**: 19,284 samples.

#### **Class 1**:
   - **Precision**: 0.76 (Lower than Class 0, more false positives).
   - **Recall**: 0.94 (High, most true instances were identified).
   - **F1-Score**: 0.84 (Balanced but slightly weaker than Class 0).
   - **Support**: 19,557 samples.

#### **Class 2**:
   - **Precision**: 0.93 (High precision, few false positives).
   - **Recall**: 0.76 (Lower recall, many true instances missed).
   - **F1-Score**: 0.83 (Good but weaker recall drags the score down).
   - **Support**: 17,882 samples.

#### **Overall Metrics**:
   - **Accuracy**: 0.87 (87% of all predictions were correct).
   - **Macro Average**:
     - Precision: 0.89 (Average precision across all classes, treating each class equally).
     - Recall: 0.87 (Average recall across all classes).
     - F1-Score: 0.87 (Average F1-score across all classes).
   - **Weighted Average**:
     - Precision: 0.89 (Weighted by the number of samples in each class).
     - Recall: 0.87 (Weighted by the number of samples in each class).
     - F1-Score: 0.88 (Balances the impact of imbalanced class sizes).

---

### **Confusion Matrix**

The confusion matrix provides a detailed breakdown of how predictions were classified:

\[
\text{Confusion Matrix} =
\begin{bmatrix}
17753 & 1461 & 70 \\
213 & 18319 & 1025 \\
32 & 4332 & 13518
\end{bmatrix}
\]

- **Row**: Represents the actual classes.
- **Column**: Represents the predicted classes.
- **Diagonal Elements**: Correctly classified samples for each class.
- **Off-Diagonal Elements**: Misclassifications.

#### Analysis:
1. **Class 0**:
   - Correctly predicted: 17,753 samples.
   - Misclassified as Class 1: 1,461 samples.
   - Misclassified as Class 2: 70 samples.

2. **Class 1**:
   - Correctly predicted: 18,319 samples.
   - Misclassified as Class 0: 213 samples.
   - Misclassified as Class 2: 1,025 samples.

3. **Class 2**:
   - Correctly predicted: 13,518 samples.
   - Misclassified as Class 0: 32 samples.
   - Misclassified as Class 1: 4,332 samples (a notable issue).

---

### **Key Observations**:
1. **Class Imbalance**:
   - Classes have similar numbers of samples, so imbalance isn’t significant.
   - However, misclassifications for Class 2 are relatively high, as seen in the confusion matrix (4,332 samples misclassified as Class 1).

2. **Class 1 Performance**:
   - Recall is high at 0.94, meaning most instances of Class 1 are detected.
   - Precision is lower at 0.76, indicating that a significant number of samples are falsely classified as Class 1.

3. **Class 2 Performance**:
   - Precision is high (0.93), but recall is relatively low (0.76), meaning many true instances of Class 2 are missed (significant misclassification as Class 1).

4. **Overall Accuracy**:
   - An accuracy of 87% is strong for multi-class classification, indicating the model performs well overall.

5. **Improvements Needed**:
   - Focus on improving Class 2 recall to reduce misclassification into Class 1.
   - Investigate potential feature engineering or model tuning to handle edge cases better.

---

This analysis indicates that while the model performs well overall, addressing recall issues for Class 2 and balancing precision for Class 1 can further enhance performance.