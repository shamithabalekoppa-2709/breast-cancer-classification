
"""
k-Nearest Neighbors Classifier model definition.

This module creates a preprocessing and classification pipeline for the
Breast Cancer Wisconsin Diagnostic dataset.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_knn_model(number_of_neighbors=5):
    """
    Create a k-Nearest Neighbors classification pipeline.

    StandardScaler is included because kNN calculates distances between
    observations and is therefore sensitive to differences in feature scale.

    Parameters
    ----------
    number_of_neighbors : int, default=5
        Number of nearest training observations used for classification.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline containing StandardScaler and KNeighborsClassifier.
    """

    knn_pipeline = Pipeline(
        steps=[
            (
                "feature_scaling",
                StandardScaler()
            ),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=number_of_neighbors,
                    weights="uniform",
                    metric="minkowski",
                    p=2
                )
            )
        ]
    )

    return knn_pipeline
