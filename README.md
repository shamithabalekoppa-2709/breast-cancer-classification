\# Breast Cancer Classification Model Explorer



\## Project Overview



This project implements and compares multiple machine-learning classification

models using the Breast Cancer Wisconsin Diagnostic dataset from the UCI Machine

Learning Repository.



An interactive Streamlit web application allows users to upload test data, select

a classification model, and view the evaluation metrics, confusion matrix,

classification report, and prediction details.



The project demonstrates an end-to-end machine-learning workflow covering data

preparation, model development, evaluation, Streamlit application development,

GitHub version control, and Streamlit Community Cloud deployment.



> \*\*Disclaimer:\*\* This project is intended only for academic and educational

> purposes. The model results must not be interpreted as medical advice or used

> as a clinical diagnostic system.



\---



\## A. Problem Statement



The objective of this project is to classify breast-mass observations as either

benign or malignant using numerical features derived from digitized images of

fine-needle aspirates of breast masses.



The following five machine-learning classification models were implemented on

the same dataset:



1\. Logistic Regression

2\. Decision Tree Classifier

3\. k-Nearest Neighbors Classifier

4\. Gaussian Naive Bayes Classifier

5\. Random Forest Classifier



Each model was evaluated using the following mandatory metrics:



\- Accuracy

\- Area Under the ROC Curve, or AUC

\- Precision

\- Recall

\- F1 Score

\- Matthews Correlation Coefficient, or MCC



The final model was selected by comparing performance across all six metrics

instead of relying only on Accuracy.



\---



\## B. Dataset Description



\### Dataset Name



Breast Cancer Wisconsin Diagnostic Dataset



\### Dataset Source



UCI Machine Learning Repository



\### Dataset Link



https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic



\### Classification Type



Binary classification



\### Dataset Details



\- Total records: 569

\- Total input features: 30

\- Target variable: `diagnosis`

\- Missing feature values: None

\- Feature type: Continuous numerical measurements

\- Positive class: Malignant



\### Target Classes



| Original Label | Encoded Value | Diagnosis |

|---|---:|---|

| B | 0 | Benign |

| M | 1 | Malignant |



Malignant was treated as the positive class while calculating Precision, Recall,

F1 Score, AUC, and related evaluation results.



\### Feature Description



The dataset contains measurements calculated from digitized images of cell

nuclei. The measurements include:



\- Radius

\- Texture

\- Perimeter

\- Area

\- Smoothness

\- Compactness

\- Concavity

\- Concave points

\- Symmetry

\- Fractal dimension



Each measurement is represented using mean, standard-error, and worst-value

variations, resulting in 30 input features.



\### Dataset Selection Justification



The dataset satisfies the assignment requirements of at least 500 instances and

12 input features. It contains 569 records and 30 continuous numerical input

features.



The binary target supports straightforward calculation and interpretation of

Accuracy, AUC, Precision, Recall, F1 Score, MCC, the confusion matrix, and the

classification report.



The dataset does not contain missing feature values and is suitable for all five

classification models implemented in this project.



\---



\## C. GitHub Repository Link



GitHub Repository:



\*\*<https://github.com/shamithabalekoppa-2709/breast-cancer-classification>\*\*



Expected format:



```text

https://github.com/<github-username>/breast-cancer-classification

```



The repository contains:



\- Complete source code

\- Model implementation files

\- Model-training notebook

\- `requirements.txt`

\- `README.md`

\- Test data

\- Original downloaded dataset

\- Streamlit application



\---



\## D. Models Used



The following five classification models were implemented using the same training

and testing datasets:



1\. Logistic Regression

2\. Decision Tree Classifier

3\. k-Nearest Neighbors Classifier

4\. Gaussian Naive Bayes Classifier

5\. Random Forest Classifier, used as the ensemble model



All models were evaluated using the same test records to ensure a fair comparison.



\### Model Comparison Table


| ML Model Name            |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-------------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression      |     0.9737 | 0.9974 |      0.9756 |   0.9524 | 0.9639 | 0.9433 |
| Decision Tree            |     0.9386 | 0.9107 |      0.973  |   0.8571 | 0.9114 | 0.8688 |
| kNN                      |     0.9737 | 0.9714 |      1      |   0.9286 | 0.963  | 0.9442 |
| Naive Bayes              |     0.9386 | 0.9897 |      0.9487 |   0.881  | 0.9136 | 0.8675 |
| Random Forest (Ensemble) |     0.9561 | 0.9947 |      0.9512 |   0.9286 | 0.9398 | 0.9054 |

All results are rounded to four decimal places for display. The original

unrounded values were used during evaluation and winner selection.



\### Observations on Model Performance



| ML Model Name                  | Observation about Model Performance                                                                                                                                                                                                                                                                                                        |
|:-------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression            | Logistic Regression achieved an Accuracy of 0.9737, AUC of 0.9974, Precision of 0.9756, Recall of 0.9524, F1 Score of 0.9639, and MCC of 0.9433. Feature standardization helped the model process measurements with different numerical scales. The model produced 2 false-negative prediction(s).                                         |
| Decision Tree                  | Decision Tree achieved an Accuracy of 0.9386, AUC of 0.9107, Precision of 0.9730, Recall of 0.8571, F1 Score of 0.9114, and MCC of 0.8688. The classifier learned threshold-based rules without feature scaling. Tree-depth and leaf-size restrictions were used to reduce overfitting. The model produced 6 false-negative prediction(s). |
| kNN                            | kNN achieved an Accuracy of 0.9737, AUC of 0.9714, Precision of 1.0000, Recall of 0.9286, F1 Score of 0.9630, and MCC of 0.9442. StandardScaler was used because kNN calculates distances between observations. The model used five nearest neighbors and produced 3 false-negative prediction(s).                                         |
| Naive Bayes                    | Gaussian Naive Bayes achieved an Accuracy of 0.9386, AUC of 0.9897, Precision of 0.9487, Recall of 0.8810, F1 Score of 0.9136, and MCC of 0.8675. The model was computationally efficient, although the conditional-independence assumption may be restrictive. The model produced 5 false-negative prediction(s).                         |
| Random Forest (Ensemble)       | Random Forest achieved an Accuracy of 0.9561, AUC of 0.9947, Precision of 0.9512, Recall of 0.9286, F1 Score of 0.9398, and MCC of 0.9054. The ensemble combined predictions from 200 Decision Trees, improving stability and capturing nonlinear relationships. The model produced 3 false-negative prediction(s).                        |
| Overall Winner for the Dataset | Logistic Regression was selected as the overall winner based on combined performance across Accuracy, AUC, Precision, Recall, F1 Score, and MCC. The model achieved an average metric rank of 1.33 and produced 2 false-negative prediction(s). The winner was selected using multiple evaluation measures rather than Accuracy alone.     |

\### Overall Winner Selection Method



The overall winner was not selected using Accuracy alone. Every model was ranked

using the six mandatory evaluation metrics:



1\. Accuracy

2\. AUC

3\. Precision

4\. Recall

5\. F1 Score

6\. MCC



A rank of 1 was assigned to the model with the highest result for a metric. The

average of the six metric ranks was calculated for every model. The model with the

lowest average rank was selected as the overall winner.



If two models had the same average rank, the following tie-breaking order was used:



1\. Higher MCC

2\. Higher F1 Score

3\. Higher AUC

4\. Higher Recall

5\. Lower false-negative count



\### Why Multiple Evaluation Metrics Were Used



Accuracy measures overall correctness, but it does not independently show how

well the model identifies malignant observations.



Precision measures the reliability of malignant predictions. Recall measures the

proportion of actual malignant observations identified correctly. F1 Score

balances Precision and Recall.



AUC evaluates the model's ability to separate the two classes across different

classification thresholds. MCC provides a balanced assessment using true

positives, true negatives, false positives, and false negatives.



Therefore, all six metrics were considered when comparing the models and selecting

the overall winner.



\---



\## Live Streamlit Application Link



Live Streamlit Application:



\*\*<PASTE STREAMLIT APPLICATION LINK HERE>\*\*



Expected format:



```text

https://<application-name>.streamlit.app

```



The application supports:



\- Uploading `test\_data.csv`

\- Selecting any one of the five models

\- Displaying all six mandatory evaluation metrics

\- Displaying a confusion matrix

\- Displaying a classification report

\- Displaying actual and predicted diagnosis values

\- Displaying malignant-class probabilities



\---



\## Project Structure



```text

breast-cancer-classification/

│

├── app.py

├── README.md

├── requirements.txt

├── test\_data.csv

├── model\_training.ipynb

│

├── data/

│   └── breast\_cancer\_wisconsin.csv

│

└── model/

&#x20;   ├── \_\_init\_\_.py

&#x20;   ├── logistic\_regression.py

&#x20;   ├── decision\_tree.py

&#x20;   ├── knn.py

&#x20;   ├── naive\_bayes.py

&#x20;   └── random\_forest.py

```



\### File Descriptions



\- `app.py`: Contains the Streamlit web application.

\- `README.md`: Contains complete project documentation.

\- `requirements.txt`: Contains the Python dependencies.

\- `test\_data.csv`: Contains the test records used in the application.

\- `model\_training.ipynb`: Contains dataset preparation, training, evaluation,

&#x20; model comparison, and winner selection.

\- `data/breast\_cancer\_wisconsin.csv`: Contains the downloaded UCI dataset.

\- `model/logistic\_regression.py`: Defines the Logistic Regression pipeline.

\- `model/decision\_tree.py`: Defines the Decision Tree model.

\- `model/knn.py`: Defines the kNN pipeline.

\- `model/naive\_bayes.py`: Defines the Gaussian Naive Bayes model.

\- `model/random\_forest.py`: Defines the Random Forest model.

\- `model/\_\_init\_\_.py`: Makes the model folder an importable package.



\---



\## Data Preparation



The following data-preparation steps were performed:



1\. Loaded the locally downloaded UCI dataset.

2\. Assigned the official column names.

3\. Verified the dataset dimensions.

4\. Removed the non-predictive `id` column.

5\. Encoded benign as `0` and malignant as `1`.

6\. Checked for missing values and duplicate records.

7\. Separated the input features and target variable.

8\. Created an 80 percent training and 20 percent testing split.

9\. Used stratified sampling to preserve class proportions.

10\. Used a fixed random state of `17` for reproducibility.

11\. Exported the test subset as `test\_data.csv`.



\### Train-Test Split



\- Training records: 455

\- Testing records: 114

\- Training percentage: Approximately 80 percent

\- Testing percentage: Approximately 20 percent

\- Stratified split: Yes

\- Random state: 17



\---



\## Streamlit Application Features



\### Dataset Upload



The evaluator can upload the provided `test\_data.csv` file.



\### Model Selection



The application provides a dropdown containing:



\- Logistic Regression

\- Decision Tree

\- kNN

\- Naive Bayes

\- Random Forest



\### Evaluation Metrics



The application displays:



\- Accuracy

\- AUC

\- Precision

\- Recall

\- F1 Score

\- MCC



\### Confusion Matrix



The application displays:



\- True negatives

\- False positives

\- False negatives

\- True positives



\### Classification Report



The application displays class-specific:



\- Precision

\- Recall

\- F1 Score

\- Support



\### Prediction Details



The application displays:



\- Actual diagnosis

\- Predicted diagnosis

\- Malignant probability

\- Whether the prediction was correct



\---



\## Installation and Local Execution



\### Prerequisites



\- Python 3.11 or a compatible version

\- pip package manager



\### Clone the Repository



```bash

git clone <PASTE GITHUB REPOSITORY LINK HERE>

```



Move into the project folder:



```bash

cd breast-cancer-classification

```



\### Install Dependencies



```bash

python -m pip install -r requirements.txt

```



\### Launch the Application



```bash

python -m streamlit run app.py

```



The application will normally open at:



```text

http://localhost:8501

```



\### Application Usage



1\. Open the Streamlit application.

2\. Select a model from the sidebar.

3\. Upload `test\_data.csv`.

4\. Review all six evaluation metrics.

5\. Open the Confusion Matrix tab.

6\. Open the Classification Report tab.

7\. Open the Predictions tab.

8\. Repeat the evaluation for the remaining models.



\---



\## Dependencies



The project requires:



\- Streamlit

\- scikit-learn

\- pandas

\- NumPy

\- Matplotlib



The dependency list is maintained in `requirements.txt`.



\---



\## Reproducibility



The following measures support reproducibility:



\- Fixed random state of `17`

\- Same train-test split for all five models

\- Stratified sampling

\- Same test dataset for every model

\- Scaling applied inside model pipelines where required

\- Same evaluation definitions across models

\- Local copy of the official UCI dataset



\---



\## BITS Virtual Lab Execution



The model-development notebook and assignment execution were performed in the

BITS Virtual Lab environment.



One screenshot showing assignment execution in the BITS Virtual Lab is included

in the final submitted PDF as required.



\---



\## Limitations



\- The dataset contains 569 records.

\- The experiment uses one fixed train-test split.

\- Model performance may change if the split or hyperparameters are changed.

\- Some input features may be correlated.

\- Cross-validation-based hyperparameter tuning was not performed.

\- The application is intended only for educational demonstration.

\- The models must not be used for medical diagnosis or treatment decisions.



\---



\## Author



\*\*Name:\*\* Shamitha B C



\*\*Program:\*\* M.Tech in Artificial Intelligence and Machine Learning



\*\*Institution:\*\* BITS Pilani, Work Integrated Learning Programme



\---



\## References



1\. UCI Machine Learning Repository, Breast Cancer Wisconsin Diagnostic Dataset:  

&#x20;  https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic



2\. scikit-learn Documentation:  

&#x20;  https://scikit-learn.org/stable/



3\. Streamlit Documentation:  

&#x20;  https://docs.streamlit.io/

