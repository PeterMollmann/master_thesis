import numpy as np
from itertools import combinations
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit


def fittingFunction(X, *coefficients, degrees=[2, 2], include_interactions=True):
    """A fitting function for 3D data with the form z = a + b*x1 + c*x2 + d*x1^2 + e*x2^2 + f*x1*x2.
    Takes the feature data in an array X, and the parameters a, b, c, d, e, f.
    Currently only works with two features.
    Args:
        X (array-like): An array containing the feature data, ex. x1 and x2. Has structure [n_feature, n_samples]
        coefficients (tuple): A tuple containing the coefficients ex. a, b, c, d, e, f.
        degrees (list): The degree of fit for each feature of the polynomial fit. First entry for the first feature and so on
        include_interactions (bool): Whether to include interaction terms, e.g. x1*x2.
    Returns:
        array-like: The fitted values.
    """
    assert len(degrees) == X.shape[0]
    n_features = X.shape[0]

    coefficient_number = 0
    y = coefficients[coefficient_number]
    coefficient_number += 1

    if degrees[0] >= 1:
        y += np.dot(coefficients[coefficient_number], X[0, :])
        coefficient_number += 1

    if degrees[1] >= 1:
        y += np.dot(coefficients[coefficient_number], X[1, :])
        coefficient_number += 1

    if degrees[0] >= 2:
        y += np.dot(coefficients[coefficient_number], X[0, :] ** 2)
        coefficient_number += 1

    if degrees[1] >= 2:
        y += np.dot(coefficients[coefficient_number], X[1, :] ** 2)
        coefficient_number += 1

    if include_interactions:
        y += np.dot(coefficients[coefficient_number], X[0, :] * X[1, :])

    return y.flatten()


def SimpleRegression(features, target, degrees=[2, 2], use_interactions: bool = True):
    """
    Perform polynomial regression on the provided data using the specified regression degree.

    Args:
        features (array-like): A tuple containing the feature data (x1, x2).
        target (array-like): The target variable data (y).
        degree (str): The regression method to use. Options are 1, 2, or 3.
        use_interactions (bool): Whether to include interaction terms, e.g. x1*x2.
    Returns:
        tuple: A tuple containing the optimal parameters, the R^2 score, and the mean square error.
    """
    x = features
    y = target

    n_features = features.shape[0]
    n_interactions = int(n_features * (n_features - 1) / 2)

    n_params = 1

    if degrees[0] >= 1:
        n_params += 1
    if degrees[1] >= 1:
        n_params += 1
    if degrees[0] >= 2:
        n_params += 1
    if degrees[1] >= 2:
        n_params += 1
    # if degrees[0] >= 2 and degrees[1] >= 2 and use_interactions:
    #     n_params += n_interactions
    if use_interactions:
        n_params += 1

    p0 = (1,) * n_params

    def model(x, *coefficients):
        return fittingFunction(
            x, *coefficients, degrees=degrees, include_interactions=use_interactions
        )

    popt, _ = curve_fit(model, x, y, p0=p0)
    y_pred = model(x, *popt)
    r2 = r2_score(y_true=y, y_pred=y_pred)
    mse = mean_squared_error(y_true=y, y_pred=y_pred)
    return popt, r2, mse
