
"""
Logistic Regression model definition.

This module creates a Logistic Regression classification pipeline for the
Breast Cancer Wisconsin Diagnostic dataset.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_logistic_regression_model(random_state=17):
    """
    Create a Logistic Regression classification pipeline.

    StandardScaler is included because the input features have different
    numerical scales. The scaler and classifier are placed in the same
    pipeline to prevent data leakage and maintain consistent preprocessing.

    Parameters
    ----------
    random_state : int, default=17
        Fixed random seed used for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline containing StandardScaler and LogisticRegression.
    """

    model_pipeline = Pipeline(
        steps=[
            (
                "feature_scaling",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=1000,
                    random_state=random_state
                )
            )
        ]
    )

    return model_pipeline
