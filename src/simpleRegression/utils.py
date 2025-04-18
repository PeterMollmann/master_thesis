import matplotlib.pyplot as plt
import numpy as np
from simpleRegression.fitFunctions import fittingFunction


def printMethodInfo(method_name, best_params, r2_score, mse):
    print("=" * 50)
    print(f"Method: {method_name}")
    print(f"Best Parameters: {best_params}")
    print(f"R^2 Score: {r2_score}")
    print(f"Mean Squared Error: {mse}")
    print("=" * 50)


def plot_data(features, target, method_name, best_params, include_interactions=True):
    """
    Plot the data and the fitted curve based on the regression method.

    Args:
        features (tuple): A tuple containing the feature data (x1, x2).
        target (array-like): The target variable data (y).
        best_params (array-like): The optimal parameters from the regression.
        method_name (str): The name of the regression method used.
    """
    if len(features) == 2:
        x1, x2 = features
        y = target

    elif len(features) == 3:
        x1, x2, x3 = features
        y = target
        x3_grid = np.array(x3).reshape(len(np.unique(x1)), len(np.unique(x2)))

    # Create a grid for plotting
    x1_grid = np.array(x1).reshape(len(np.unique(x1)), len(np.unique(x2)))
    x2_grid = np.array(x2).reshape(len(np.unique(x1)), len(np.unique(x2)))
    y_true_grid = np.array(y).reshape(len(np.unique(x1)), len(np.unique(x2)))
    y_grid = None

    if method_name == "linear":
        degree = 1
    elif method_name == "secondOrder":
        degree = 2

    # Calculate the fitted values based on the regression method
    y_grid = fittingFunction(
        features, *best_params, degree=degree, include_interactions=include_interactions
    )

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x1, x2, y, color="r", label="Data Points")
    ax.plot_surface(
        x1_grid, x2_grid, y_true_grid, alpha=0.5, color="g", label="True Surface"
    )
    ax.plot_surface(
        x1_grid,
        x2_grid,
        y_grid.reshape(40, 5),
        alpha=0.5,
        color="b",
        label="Fitted Surface",
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Target Variable")
    plt.title(f"{method_name} Regression Fit")
    plt.legend()
    plt.show()
