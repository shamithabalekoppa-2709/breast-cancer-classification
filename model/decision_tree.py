
"""
Decision Tree Classifier model definition.

This module creates a Decision Tree Classifier for the Breast Cancer
Wisconsin Diagnostic classification problem.
"""

from sklearn.tree import DecisionTreeClassifier


def create_decision_tree_model(random_state=17):
    """
    Create a Decision Tree classification model.

    Feature scaling is not included because a Decision Tree uses feature
    thresholds to divide the dataset and is not sensitive to differences
    in feature magnitude.

    Parameters
    ----------
    random_state : int, default=17
        Fixed random seed used for reproducibility.

    Returns
    -------
    sklearn.tree.DecisionTreeClassifier
        Configured Decision Tree classification model.
    """

    decision_tree_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state
    )

    return decision_tree_model
