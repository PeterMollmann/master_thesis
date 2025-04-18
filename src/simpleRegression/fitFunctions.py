import numpy as np
from itertools import combinations
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit


def fittingFunction(X, *coefficients, degree=2, include_interactions=True):
    """A fitting function for 3D data with the form z = a + b*x1 + c*x2 + d*x1^2 + e*x2^2 + f*x1*x2.
    Takes the feature data in an array X, and the parameters a, b, c, d, e, f.
    Args:
        X (array-like): A tuple containing the feature data, ex. x1 and x2.
        coefficients (tuple): A tuple containing the coefficients ex. a, b, c, d, e, f.
        degree (int): The degree of the polynomial fit.
        include_interactions (bool): Whether to include interaction terms, e.g. x1*x2.
    Returns:
        array-like: The fitted values.
    """
    n_features = X.shape[0]

    y = coefficients[0]

    if degree >= 1:
        y += np.dot(coefficients[1 : n_features + 1], X)

    if degree >= 2:
        y += np.dot(coefficients[n_features + 1 : 2 * n_features + 1], X**2)

    if degree >= 2 and include_interactions:
        n_interactions = int(n_features * (n_features - 1) / 2)
        interaction_coefficients = coefficients[
            2 * n_features + 1 : 2 * n_features + 1 + n_interactions
        ]
        for dk, (i, j) in zip(
            interaction_coefficients, combinations(range(n_features), 2)
        ):
            y += dk * X[i] * X[j]

    return y.flatten()


def SimpleRegression(features, target, degree=1, use_interactions: bool = True):
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
    n_interactions = int(n_features * (n_features - 1) / 2) if use_interactions else 0

    n_params = 0

    if degree >= 1:
        n_params = 1 + n_features
    if degree >= 2:
        n_params += n_features
    if degree >= 2 and use_interactions:
        n_params += n_interactions

    p0 = (1,) * n_params

    def model(x, *coefficients):
        return fittingFunction(
            x, *coefficients, degree=degree, include_interactions=use_interactions
        )

    popt, _ = curve_fit(model, x, y, p0=p0)
    y_pred = model(x, *popt)
    r2 = r2_score(y_true=y, y_pred=y_pred)
    mse = mean_squared_error(y_true=y, y_pred=y_pred)
    return popt, r2, mse
