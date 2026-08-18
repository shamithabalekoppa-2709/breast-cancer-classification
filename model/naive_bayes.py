
"""
Gaussian Naive Bayes model definition.

This module creates a Gaussian Naive Bayes classifier for the Breast Cancer
Wisconsin Diagnostic classification problem.
"""

from sklearn.naive_bayes import GaussianNB


def create_naive_bayes_model():
    """
    Create a Gaussian Naive Bayes classification model.

    Gaussian Naive Bayes is used because the input features are continuous
    numerical measurements. Feature scaling is not mandatory for this model
    because GaussianNB estimates a mean and variance for each feature within
    each target class.

    Returns
    -------
    sklearn.naive_bayes.GaussianNB
        Configured Gaussian Naive Bayes classifier.
    """

    naive_bayes_model = GaussianNB(
        var_smoothing=1e-9
    )

    return naive_bayes_model
