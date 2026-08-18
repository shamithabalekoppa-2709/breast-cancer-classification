from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

from model.decision_tree import create_decision_tree_model
from model.knn import create_knn_model
from model.logistic_regression import create_logistic_regression_model
from model.naive_bayes import create_naive_bayes_model
from model.random_forest import create_random_forest_model


# ------------------------------------------------------------
# Streamlit page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_DIRECTORY = Path(__file__).resolve().parent

TRAINING_DATA_PATH = (
    PROJECT_DIRECTORY
    / "data"
    / "breast_cancer_wisconsin.csv"
)


# ------------------------------------------------------------
# Dataset column names
# ------------------------------------------------------------

UCI_COLUMN_NAMES = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]


FEATURE_NAMES = [
    column_name
    for column_name in UCI_COLUMN_NAMES
    if column_name not in ["id", "diagnosis"]
]


MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest"
]


# ------------------------------------------------------------
# Load and prepare training data
# ------------------------------------------------------------

@st.cache_data
def load_and_split_training_data():
    """
    Load the locally stored UCI dataset and reproduce the
    train-test split used in the model-training notebook.

    Returns
    -------
    tuple
        X_train and y_train.
    """

    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            "The training dataset was not found at "
            "data/breast_cancer_wisconsin.csv."
        )

    dataset_df = pd.read_csv(
        TRAINING_DATA_PATH,
        header=None,
        names=UCI_COLUMN_NAMES
    )

    if dataset_df.shape != (569, 32):
        raise ValueError(
            "The training dataset has an unexpected shape. "
            f"Expected (569, 32), but received {dataset_df.shape}."
        )

    dataset_df = dataset_df.drop(
        columns=["id"]
    )

    dataset_df["diagnosis"] = (
        dataset_df["diagnosis"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({
            "B": 0,
            "M": 1
        })
    )

    if dataset_df["diagnosis"].isnull().any():
        raise ValueError(
            "Unexpected values were found in the training "
            "diagnosis column."
        )

    feature_data = dataset_df.drop(
        columns=["diagnosis"]
    )

    target_data = dataset_df["diagnosis"].astype(int)

    X_train, _, y_train, _ = train_test_split(
        feature_data,
        target_data,
        test_size=0.20,
        random_state=17,
        stratify=target_data
    )

    return X_train, y_train


# ------------------------------------------------------------
# Create and train all five models
# ------------------------------------------------------------

@st.cache_resource
def train_all_models():
    """
    Create and train all five classification models.

    Returns
    -------
    dict
        Dictionary containing the trained models.
    """

    X_train, y_train = load_and_split_training_data()

    trained_models = {
        "Logistic Regression":
            create_logistic_regression_model(
                random_state=17
            ),

        "Decision Tree":
            create_decision_tree_model(
                random_state=17
            ),

        "kNN":
            create_knn_model(
                number_of_neighbors=5
            ),

        "Naive Bayes":
            create_naive_bayes_model(),

        "Random Forest":
            create_random_forest_model(
                random_state=17
            )
    }

    for model_name, model in trained_models.items():
        model.fit(
            X_train,
            y_train
        )

    return trained_models


# ------------------------------------------------------------
# Validate uploaded test data
# ------------------------------------------------------------

def validate_uploaded_data(uploaded_df):
    """
    Validate the uploaded test CSV.

    Parameters
    ----------
    uploaded_df : pandas.DataFrame
        Test data uploaded through Streamlit.

    Returns
    -------
    tuple
        Validation status and validation message.
    """

    required_columns = FEATURE_NAMES + ["diagnosis"]

    if uploaded_df.empty:
        return (
            False,
            "The uploaded CSV does not contain any records."
        )

    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in uploaded_df.columns
    ]

    if missing_columns:
        return (
            False,
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    unexpected_columns = [
        column_name
        for column_name in uploaded_df.columns
        if column_name not in required_columns
    ]

    if unexpected_columns:
        return (
            False,
            "Unexpected columns found: "
            + ", ".join(unexpected_columns)
        )

    if uploaded_df[required_columns].isnull().sum().sum() > 0:
        return (
            False,
            "The uploaded CSV contains missing values."
        )

    diagnosis_values = set(
        uploaded_df["diagnosis"]
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
    )

    valid_diagnosis_values = {
        "B",
        "M",
        "0",
        "1"
    }

    if not diagnosis_values.issubset(
        valid_diagnosis_values
    ):
        return (
            False,
            "The diagnosis column must contain B and M, "
            "or encoded values 0 and 1."
        )

    non_numeric_features = []

    for column_name in FEATURE_NAMES:
        converted_column = pd.to_numeric(
            uploaded_df[column_name],
            errors="coerce"
        )

        if converted_column.isnull().any():
            non_numeric_features.append(
                column_name
            )

    if non_numeric_features:
        return (
            False,
            "The following feature columns contain "
            "non-numerical values: "
            + ", ".join(non_numeric_features)
        )

    if len(uploaded_df) < 2:
        return (
            False,
            "The uploaded test CSV must contain at least "
            "two records."
        )

    return (
        True,
        "Test data validation completed successfully."
    )


# ------------------------------------------------------------
# Encode the uploaded target
# ------------------------------------------------------------

def encode_uploaded_target(diagnosis_series):
    """
    Convert B, M, 0, and 1 labels into integer values.

    Parameters
    ----------
    diagnosis_series : pandas.Series
        Diagnosis values from uploaded test data.

    Returns
    -------
    pandas.Series
        Encoded diagnosis values.
    """

    cleaned_diagnosis = (
        diagnosis_series
        .astype(str)
        .str.strip()
        .str.upper()
    )

    encoded_diagnosis = cleaned_diagnosis.map({
        "B": 0,
        "M": 1,
        "0": 0,
        "1": 1
    })

    return encoded_diagnosis.astype(int)


# ------------------------------------------------------------
# Calculate model evaluation results
# ------------------------------------------------------------

def calculate_model_results(
    selected_model,
    X_test,
    y_test
):
    """
    Generate model predictions and calculate all six
    mandatory evaluation metrics.

    Parameters
    ----------
    selected_model
        Trained scikit-learn classification model.

    X_test : pandas.DataFrame
        Uploaded test features.

    y_test : pandas.Series
        Encoded test target.

    Returns
    -------
    tuple
        Predictions, malignant probabilities, and metrics.
    """

    predictions = selected_model.predict(
        X_test
    )

    malignant_probabilities = (
        selected_model
        .predict_proba(X_test)[:, 1]
    )

    evaluation_metrics = {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "AUC": roc_auc_score(
            y_test,
            malignant_probabilities
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test,
            predictions
        )
    }

    return (
        predictions,
        malignant_probabilities,
        evaluation_metrics
    )


# ------------------------------------------------------------
# Application header
# ------------------------------------------------------------

st.title(
    "Breast Cancer Classification Model Explorer"
)

st.write(
    "Upload the provided test_data.csv file and select a "
    "classification model to view its evaluation results."
)

st.info(
    "This application is an educational machine-learning "
    "demonstration and is not a clinical diagnostic system."
)


# ------------------------------------------------------------
# Sidebar model-selection controls
# ------------------------------------------------------------

with st.sidebar:
    st.header("Model Configuration")

    selected_model_name = st.selectbox(
        "Select a classification model",
        options=MODEL_NAMES
    )

    st.subheader("Expected Test Data")

    st.write(
        "The uploaded CSV must contain 30 numerical feature "
        "columns and one diagnosis column."
    )

    st.write(
        "Diagnosis values may be B and M, or 0 and 1."
    )

    st.write(
        "Use the test_data.csv file included in the "
        "GitHub repository."
    )


# ------------------------------------------------------------
# Upload the test CSV
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload test data",
    type=["csv"],
    help=(
        "Upload the test_data.csv file generated by "
        "the model-training notebook."
    )
)


if uploaded_file is None:
    st.warning(
        "Upload test_data.csv to begin model evaluation."
    )

    st.stop()


# ------------------------------------------------------------
# Read the uploaded CSV
# ------------------------------------------------------------

try:
    uploaded_test_df = pd.read_csv(
        uploaded_file
    )

except Exception as error:
    st.error(
        f"The uploaded CSV could not be read: {error}"
    )

    st.stop()


# ------------------------------------------------------------
# Display uploaded-data details
# ------------------------------------------------------------

st.subheader(
    "Uploaded Test Data Preview"
)

st.dataframe(
    uploaded_test_df.head(10),
    use_container_width=True
)

upload_detail_column_1, upload_detail_column_2 = st.columns(2)

upload_detail_column_1.metric(
    "Uploaded Records",
    uploaded_test_df.shape[0]
)

upload_detail_column_2.metric(
    "Uploaded Columns",
    uploaded_test_df.shape[1]
)


# ------------------------------------------------------------
# Validate the uploaded CSV
# ------------------------------------------------------------

is_valid, validation_message = validate_uploaded_data(
    uploaded_test_df
)

if not is_valid:
    st.error(
        validation_message
    )

    st.stop()

st.success(
    validation_message
)


# ------------------------------------------------------------
# Prepare the uploaded features and target
# ------------------------------------------------------------

X_uploaded_test = uploaded_test_df[
    FEATURE_NAMES
].copy()

for feature_name in FEATURE_NAMES:
    X_uploaded_test[feature_name] = pd.to_numeric(
        X_uploaded_test[feature_name],
        errors="raise"
    )


y_uploaded_test = encode_uploaded_target(
    uploaded_test_df["diagnosis"]
)


if y_uploaded_test.nunique() < 2:
    st.error(
        "AUC cannot be calculated because the uploaded test "
        "data contains only one diagnosis class."
    )

    st.stop()


# ------------------------------------------------------------
# Train or retrieve the cached models
# ------------------------------------------------------------

try:
    trained_models = train_all_models()

except Exception as error:
    st.error(
        f"The models could not be trained: {error}"
    )

    st.stop()


selected_model = trained_models[
    selected_model_name
]


# ------------------------------------------------------------
# Evaluate the selected model
# ------------------------------------------------------------

try:
    (
        uploaded_predictions,
        uploaded_probabilities,
        evaluation_metrics
    ) = calculate_model_results(
        selected_model,
        X_uploaded_test,
        y_uploaded_test
    )

except Exception as error:
    st.error(
        f"The selected model could not evaluate the "
        f"uploaded data: {error}"
    )

    st.stop()


# ------------------------------------------------------------
# Display the selected model and evaluation metrics
# ------------------------------------------------------------

st.subheader(
    f"Evaluation Results: {selected_model_name}"
)


metric_column_1, metric_column_2, metric_column_3 = (
    st.columns(3)
)

metric_column_1.metric(
    "Accuracy",
    f"{evaluation_metrics['Accuracy']:.4f}"
)

metric_column_2.metric(
    "AUC Score",
    f"{evaluation_metrics['AUC']:.4f}"
)

metric_column_3.metric(
    "Precision",
    f"{evaluation_metrics['Precision']:.4f}"
)


metric_column_4, metric_column_5, metric_column_6 = (
    st.columns(3)
)

metric_column_4.metric(
    "Recall",
    f"{evaluation_metrics['Recall']:.4f}"
)

metric_column_5.metric(
    "F1 Score",
    f"{evaluation_metrics['F1']:.4f}"
)

metric_column_6.metric(
    "MCC Score",
    f"{evaluation_metrics['MCC']:.4f}"
)


# ------------------------------------------------------------
# Create tabs for detailed results
# ------------------------------------------------------------

confusion_tab, report_tab, prediction_tab = st.tabs(
    [
        "Confusion Matrix",
        "Classification Report",
        "Predictions"
    ]
)


# ------------------------------------------------------------
# Confusion-matrix tab
# ------------------------------------------------------------

with confusion_tab:
    st.subheader(
        "Confusion Matrix"
    )

    confusion_figure, confusion_axis = plt.subplots(
        figsize=(6, 5)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_uploaded_test,
        uploaded_predictions,
        labels=[0, 1],
        display_labels=[
            "Benign",
            "Malignant"
        ],
        cmap="Blues",
        values_format="d",
        ax=confusion_axis
    )

    confusion_axis.set_title(
        f"{selected_model_name} Confusion Matrix"
    )

    confusion_axis.set_xlabel(
        "Predicted Diagnosis"
    )

    confusion_axis.set_ylabel(
        "Actual Diagnosis"
    )

    confusion_figure.tight_layout()

    st.pyplot(
        confusion_figure
    )

    plt.close(
        confusion_figure
    )

    confusion_matrix_values = confusion_matrix(
        y_uploaded_test,
        uploaded_predictions,
        labels=[0, 1]
    )

    (
        true_negatives,
        false_positives,
        false_negatives,
        true_positives
    ) = confusion_matrix_values.ravel()

    st.markdown("#### Confusion Matrix Details")

    detail_column_1, detail_column_2 = st.columns(2)

    detail_column_1.metric(
        "True Negatives",
        int(true_negatives)
    )

    detail_column_2.metric(
        "False Positives",
        int(false_positives)
    )

    detail_column_3, detail_column_4 = st.columns(2)

    detail_column_3.metric(
        "False Negatives",
        int(false_negatives)
    )

    detail_column_4.metric(
        "True Positives",
        int(true_positives)
    )

    st.write(
        "False negatives represent malignant records that "
        "were incorrectly classified as benign."
    )


# ------------------------------------------------------------
# Classification-report tab
# ------------------------------------------------------------

with report_tab:
    st.subheader(
        "Classification Report"
    )

    report_dictionary = classification_report(
        y_uploaded_test,
        uploaded_predictions,
        labels=[0, 1],
        target_names=[
            "Benign",
            "Malignant"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = (
        pd.DataFrame(report_dictionary)
        .transpose()
    )

    st.dataframe(
        report_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:.0f}"
        }),
        use_container_width=True
    )

    st.write(
        "The classification report presents precision, recall, "
        "F1 score, and support for both diagnosis classes."
    )


# ------------------------------------------------------------
# Prediction-details tab
# ------------------------------------------------------------

with prediction_tab:
    st.subheader(
        "Prediction Details"
    )

    actual_diagnosis = (
        y_uploaded_test
        .reset_index(drop=True)
        .map({
            0: "Benign",
            1: "Malignant"
        })
    )

    predicted_diagnosis = (
        pd.Series(uploaded_predictions)
        .map({
            0: "Benign",
            1: "Malignant"
        })
    )

    prediction_results_df = pd.DataFrame({
        "Actual Diagnosis": actual_diagnosis,
        "Predicted Diagnosis": predicted_diagnosis,
        "Malignant Probability": uploaded_probabilities
    })

    prediction_results_df["Prediction Correct"] = (
        prediction_results_df["Actual Diagnosis"]
        == prediction_results_df["Predicted Diagnosis"]
    )

    st.dataframe(
        prediction_results_df.style.format({
            "Malignant Probability": "{:.4f}"
        }),
        use_container_width=True
    )

    correct_prediction_count = int(
        prediction_results_df[
            "Prediction Correct"
        ].sum()
    )

    incorrect_prediction_count = int(
        (
            ~prediction_results_df[
                "Prediction Correct"
            ]
        ).sum()
    )

    prediction_column_1, prediction_column_2 = (
        st.columns(2)
    )

    prediction_column_1.metric(
        "Correct Predictions",
        correct_prediction_count
    )

    prediction_column_2.metric(
        "Incorrect Predictions",
        incorrect_prediction_count
    )


# ------------------------------------------------------------
# Application footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "Dataset: Breast Cancer Wisconsin Diagnostic, "
    "UCI Machine Learning Repository."
)

st.caption(
    "This application is developed for academic and educational "
    "purposes. The model results must not be treated as medical advice."
)
   