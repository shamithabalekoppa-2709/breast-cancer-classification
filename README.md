<h1 align="center"> Breast Cancer Classification Model Explorer</h1>

<p align="center">
  <strong>Machine Learning Classification | Model Evaluation | Streamlit Deployment</strong>
</p>

<p align="center">
  A comparative evaluation of five classification models using the
  UCI Breast Cancer Wisconsin Diagnostic Dataset.
</p>

---

##  Project Overview

This project implements and compares multiple machine-learning classification models using the **Breast Cancer Wisconsin Diagnostic Dataset** from the **UCI Machine Learning Repository**.

An interactive **Streamlit web application** allows users to:

- Upload the provided test dataset
- Select a classification model
- View all five mandatory evaluation metrics
- Examine the confusion matrix
- Review the classification report
- Compare actual and predicted diagnosis values
- View malignant-class probability scores

The project demonstrates an end-to-end machine-learning workflow covering:

- **Data preparation**
- **Model implementation**
- **Model evaluation**
- **Streamlit application development**
- **GitHub version control**
- **Streamlit Community Cloud deployment**

> **Important Disclaimer:** This project is intended only for academic and educational purposes. The model results must not be interpreted as medical advice or used as a clinical diagnostic system.

---

## A. Problem Statement

Breast cancer is a significant global health concern, and the timely and accurate
classification of breast masses can support informed clinical assessment. Machine
learning provides an opportunity to analyse quantitative measurements extracted
from digitized fine-needle aspiration images and identify patterns associated with
benign and malignant observations.

This project develops an automated, end-to-end machine-learning classification
pipeline using the **Breast Cancer Wisconsin Diagnostic Dataset** from the UCI
Machine Learning Repository. The dataset contains 569 observations and 30
continuous numerical features derived from digitized images of fine-needle
aspirates of breast masses.

The primary objective is to implement, evaluate, and compare five classification
algorithms for predicting one of two diagnostic classes:

- **Malignant:** Cancerous breast-mass observation
- **Benign:** Non-cancerous breast-mass observation

The following machine-learning models are implemented using the same training and
testing datasets:

1. Logistic Regression
2. Decision Tree Classifier
3. k-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier as the ensemble model

The following five machine-learning classification models were implemented on the same dataset:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **k-Nearest Neighbors Classifier**
4. **Gaussian Naive Bayes Classifier**
5. **Random Forest Classifier**

Each model was evaluated using the following mandatory metrics:

- **Accuracy**
- **Area Under the ROC Curve, or AUC**
- **Precision**
- **Recall**
- **F1 Score**
- **Matthews Correlation Coefficient, or MCC**

The final model was selected by comparing performance across all six metrics instead of relying only on Accuracy.

---

## B. Dataset Description

### Dataset Summary

| Attribute | Description |
|---|---|
| **Dataset Name** | Breast Cancer Wisconsin Diagnostic Dataset |
| **Dataset Source** | UCI Machine Learning Repository |
| **Classification Type** | Binary Classification |
| **Total Records** | 569 |
| **Input Features** | 30 |
| **Target Variable** | `diagnosis` |
| **Missing Feature Values** | None |
| **Feature Type** | Continuous Numerical Measurements |
| **Positive Class** | Malignant |

### Dataset Link

[View the Breast Cancer Wisconsin Diagnostic Dataset on UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

### Target Classes

| Original Label | Encoded Value | Diagnosis |
|---|---:|---|
| `B` | 0 | Benign |
| `M` | 1 | Malignant |

Malignant was treated as the positive class while calculating Precision, Recall, F1 Score, AUC, and the related evaluation results.

### Feature Description

The dataset contains measurements calculated from digitized images of cell nuclei. The measurements include:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

Each measurement is represented using mean, standard-error, and worst-value variations, resulting in 30 input features.

### Dataset Selection Justification

The dataset satisfies the assignment requirements of at least **500 instances** and **12 input features**. It contains 569 records and 30 continuous numerical input features.

The binary target supports straightforward calculation and interpretation of Accuracy, AUC, Precision, Recall, F1 Score, MCC, the confusion matrix, and the classification report.

The dataset does not contain missing feature values and is suitable for all five classification models implemented in this project.

---

## C. GitHub Repository Link

### Project Repository

https://github.com/shamithabalekoppa-2709/breast-cancer-classification

The repository contains:

- Complete source code
- Separate model implementation files
- Model-training notebook
- `requirements.txt`
- `README.md`
- Test dataset
- Original downloaded UCI dataset
- Streamlit application

---

## D. Models Used

The following five classification models were implemented using the same training and testing datasets:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **k-Nearest Neighbors Classifier**
4. **Gaussian Naive Bayes Classifier**
5. **Random Forest Classifier**, used as the ensemble model

All models were evaluated using the same test records to ensure a fair and consistent comparison.

###  Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.9737 | 0.9974 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| **Decision Tree** | 0.9386 | 0.9107 | 0.9730 | 0.8571 | 0.9114 | 0.8688 |
| **kNN** | 0.9737 | 0.9714 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| **Naive Bayes** | 0.9386 | 0.9897 | 0.9487 | 0.8810 | 0.9136 | 0.8675 |
| **Random Forest (Ensemble)** | 0.9561 | 0.9947 | 0.9512 | 0.9286 | 0.9398 | 0.9054 |

> All results are rounded to four decimal places for display. The original unrounded values were used during model evaluation and winner selection.

###  Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Logistic Regression achieved an Accuracy of 0.9737, AUC of 0.9974, Precision of 0.9756, Recall of 0.9524, F1 Score of 0.9639, and MCC of 0.9433. Feature standardization helped the model process measurements with different numerical scales. The model produced 2 false-negative predictions. |
| **Decision Tree** | Decision Tree achieved an Accuracy of 0.9386, AUC of 0.9107, Precision of 0.9730, Recall of 0.8571, F1 Score of 0.9114, and MCC of 0.8688. The classifier learned threshold-based rules without requiring feature scaling. Tree-depth and leaf-size restrictions were used to reduce overfitting. The model produced 6 false-negative predictions. |
| **kNN** | kNN achieved an Accuracy of 0.9737, AUC of 0.9714, Precision of 1.0000, Recall of 0.9286, F1 Score of 0.9630, and MCC of 0.9442. StandardScaler was used because kNN calculates distances between observations. The model used five nearest neighbors and produced 3 false-negative predictions. |
| **Naive Bayes** | Gaussian Naive Bayes achieved an Accuracy of 0.9386, AUC of 0.9897, Precision of 0.9487, Recall of 0.8810, F1 Score of 0.9136, and MCC of 0.8675. The model was computationally efficient, although the conditional-independence assumption may be restrictive because several measurements are related. The model produced 5 false-negative predictions. |
| **Random Forest (Ensemble)** | Random Forest achieved an Accuracy of 0.9561, AUC of 0.9947, Precision of 0.9512, Recall of 0.9286, F1 Score of 0.9398, and MCC of 0.9054. The ensemble combined predictions from 200 Decision Trees, improving stability and capturing nonlinear feature relationships. The model produced 3 false-negative predictions. |
| **Overall Winner for the Dataset** | Logistic Regression was selected as the overall winner based on combined performance across Accuracy, AUC, Precision, Recall, F1 Score, and MCC. The model achieved an average metric rank of 1.33 and produced 2 false-negative predictions. The winner was selected using multiple evaluation measures rather than Accuracy alone. |

###  Overall Winner Selection Method

The overall winner was not selected using Accuracy alone. Every model was ranked using the six mandatory evaluation metrics:

1. Accuracy
2. AUC
3. Precision
4. Recall
5. F1 Score
6. MCC

A rank of 1 was assigned to the model with the highest result for a metric. The average of the six metric ranks was calculated for every model. The model with the lowest average rank was selected as the overall winner.

If two models had the same average rank, the following tie-breaking order was used:

1. Higher MCC
2. Higher F1 Score
3. Higher AUC
4. Higher Recall
5. Lower false-negative count

### Overall Winner

> **Logistic Regression** was selected as the overall winner with an average metric rank of **1.33**.

Logistic Regression achieved the highest AUC, Recall, and F1 Score among the evaluated models while also producing the lowest false-negative count. Its strong and balanced performance across the six mandatory metrics supported its selection as the overall winner.

### Why Multiple Evaluation Metrics Were Used

Accuracy measures overall correctness, but it does not independently show how effectively the model identifies malignant observations.

- **Precision** measures the reliability of malignant predictions.
- **Recall** measures the proportion of actual malignant observations identified correctly.
- **F1 Score** balances Precision and Recall.
- **AUC** evaluates the model's ability to separate benign and malignant classes across different classification thresholds.
- **MCC** provides a balanced assessment using true positives, true negatives, false positives, and false negatives.

Therefore, all six metrics were considered when comparing the models and selecting the overall winner.

---

##  Live Streamlit Application

### Application Link

**The live Streamlit application link will be added after deployment.**

Replace the line above with the final link in this format:

```text
https://breast-cancer-classification-bgfbpckaywj7edfsu4dle4.streamlit.app/
