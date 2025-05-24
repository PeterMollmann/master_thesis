import matplotlib.pyplot as plt
import numpy as np
from simpleRegression.fitFunctions import fittingFunction
from sklearn.model_selection import train_test_split


def printMethodInfo(
    method_name,
    target_name,
    best_params,
    n_features: int,
    use_interactions: bool,
    train_r2_score: float,
    train_mse: float,
    test_r2_score: float = None,
    test_mse: float = None,
):
    """
    Print the regression method information, including the fitted function and parameters.
    Args:
        method_name (str): The name of the regression method used.
        best_params (array-like): The optimal parameters from the regression.
        n_features (int): The number of features used in the regression.
        use_interactions (bool): Whether interaction terms were included.
        r2_score (float): The R^2 score of the regression.
        mse (float): The mean squared error of the regression.
    """
    print("=" * 50)
    print(f"Method: {method_name}")
    print(f"Target: {target_name}")
    printFittedFunction(
        popt=best_params,
        n_features=n_features,
        degree=1 if method_name == "linear" else 2,
        use_interactions=use_interactions,
    )
    print(f"Best Parameters: {best_params}")
    print(f"Train R^2 Score: {train_r2_score}")
    print(f"Train Mean Squared Error: {train_mse}")
    print(f"Test R^2 Score: {test_r2_score}") if test_r2_score is not None else "N/A"
    print(f"Test Mean Squared Error: {test_mse}") if test_mse is not None else "N/A"
    print("=" * 50)


def plot_data(
    features_all,
    target_all,
    features_test,
    target_test,
    best_params,
    degrees: list = [2, 2],
    include_interactions: bool = True,
    xlabel: str = "Feature 1",
    ylabel: str = "Feature 2",
    zlabel: str = "Target Variable",
    target_name: str = "Target",
    random_seed: int = 42,
    save_fig: bool = False,
):
    """
    Plot the data and the fitted curve based on the regression method.

    Args:
        features (tuple): A tuple containing the feature data (x1, x2).
        target (array-like): The target variable data (y).
        method_name (str): The name of the regression method used.
        best_params (array-like): The optimal parameters from the regression.
        include_interactions (bool): Whether to include interaction terms.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        zlabel (str): The label for the z-axis.
        target_name (str): The name of the target variable.
        save_fig (bool): Whether to save the figure as a PNG file.

    """

    x1 = features_all[0, :]
    x2 = features_all[1, :]
    y = target_all
    x1_test = features_test[0, :]
    x2_test = features_test[1, :]
    y_test = target_test

    # Calculate the fitted values based on the regression method
    y_pred = fittingFunction(
        features_all,
        *best_params,
        degrees=degrees,
        include_interactions=include_interactions,
    )
    y_pred_grid = y_pred.reshape(len(np.unique(x1)), len(np.unique(x2)))

    # Create a grid for surface plotting
    x1_grid = np.array(x1).reshape(len(np.unique(x1)), len(np.unique(x2)))
    x2_grid = np.array(x2).reshape(len(np.unique(x1)), len(np.unique(x2)))
    y_grid = np.array(y).reshape(len(np.unique(x1)), len(np.unique(x2)))

    # Scatter plotting train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        features_all.T, target_all, test_size=0.2, random_state=random_seed
    )
    x1, x2 = X_train[:, 0], X_train[:, 1]
    x1_test, x2_test = X_test[:, 0], X_test[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x1,
        x2,
        y_train,
        color="r",
        linewidths=1,
        alpha=1,
        # edgecolors="k",
        label="Train Data",
    )
    ax.scatter(
        x1_test,
        x2_test,
        y_test,
        color="k",
        linewidths=1,
        alpha=1,
        # edgecolors="k",
        label="Test Data",
    )

    # Surface plotting
    ax.plot_surface(
        x1_grid,
        x2_grid,
        y_grid,
        alpha=0.5,
        color="g",
        label="FEM Surface",
    )
    ax.plot_surface(
        x1_grid,
        x2_grid,
        y_pred_grid,
        alpha=0.5,
        color="b",
        label="Fitted Surface",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    ax.set_xlim([np.min(x1), np.max(x1)])
    ax.set_ylim([np.min(x2), np.max(x2)])
    if np.min(y) > np.min(y_pred) and np.max(y) > np.max(y_pred):
        ax.set_zlim([np.min(y_pred), np.max(y)])
    elif np.min(y) < np.min(y_pred) and np.max(y) > np.max(y_pred):
        ax.set_zlim([np.min(y), np.max(y)])
    elif np.min(y) > np.min(y_pred) and np.max(y) < np.max(y_pred):
        ax.set_zlim([np.min(y_pred), np.max(y_pred)])
    elif np.min(y) < np.min(y_pred) and np.max(y) < np.max(y_pred):
        ax.set_zlim([np.min(y), np.max(y_pred)])

    # plt.title(f"{method_name} Regression Fit")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    # ax.view_init(elev=20, azim=30)
    # plt.grid()

    if save_fig:
        plt.savefig(
            "src/simpleRegression/saved_figs/"
            + str(target_name)
            + "_nFeatures"
            + str(len(features_all))
            + "_degree"
            + str(degrees)
            + "_interactions"
            + str(include_interactions)
            + ".png"
        )
    else:
        plt.show()
    plt.close(fig)


def printFittedFunction(
    popt, n_features: int = 2, degree: int = 2, use_interactions: bool = True
):
    """
    Print the fitted function based on the regression method and parameters.
    Args:
        popt (array-like): The optimal parameters from the regression.
        n_features (int): The number of features used in the regression.
        degree (int): The degree of the polynomial fit.
        use_interactions (bool): Whether to include interaction terms, e.g. x1*x2.
    """

    if degree == 1:
        print("Linear regression")
        if n_features == 2:
            print(f"y = {popt[0]} + {popt[1]} * x1 + {popt[2]} * x2")
        elif n_features == 3:
            print(f"y = {popt[0]} + {popt[1]} * x1 + {popt[2]} * x2 + {popt[3]} * x3")
    elif degree == 2:
        print("Polynomial regression")
        if use_interactions:
            print("Including interaction terms")
            if n_features == 2:
                print(
                    f"y = {popt[0]} + {popt[1]} * x1 + {popt[2]} * x2 + {popt[3]} * x1^2 + {popt[4]} * x2^2 + {popt[5]} * x1*x2"
                )
            elif n_features == 3:
                print(
                    f"y = {popt[0]} + {popt[1]} * x1 + {popt[2]} * x2 + {popt[3]} * x3 + {popt[4]} * x1^2 + {popt[5]} * x2^2 + {popt[6]} * x3^2 + {popt[7]} * x1*x2 + {popt[8]} * x1*x3 + {popt[9]} * x2*x3"
                )
        else:
            if n_features == 2:
                print(
                    f"y = {popt[0]} + {popt[1]} * x1 + {popt[2]} * x2 + {popt[3]} * x1^2 + {popt[4]} * x2^2"
                )
            elif n_features == 3:
                print(
                    f"y = {popt[0]} + {popt[1]} * x1 + {popt[2]} * x2 + {popt[3]} * x3 + {popt[4]} * x1^2 + {popt[5]} * x2^2 + {popt[6]} * x3^2"
                )
