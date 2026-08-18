
"""
Random Forest Classifier model definition.

This module creates a Random Forest ensemble classifier for the Breast Cancer
Wisconsin Diagnostic classification problem.
"""

from sklearn.ensemble import RandomForestClassifier


def create_random_forest_model(random_state=17):
    """
    Create a Random Forest classification model.

    Random Forest combines predictions from multiple Decision Trees.
    Feature scaling is not required because the model uses feature-based
    threshold splits rather than distance calculations.

    Parameters
    ----------
    random_state : int, default=17
        Fixed random seed used for reproducibility.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Configured Random Forest classification model.
    """

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1
    )

    return random_forest_model
