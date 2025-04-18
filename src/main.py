import numpy as np
import pandas as pd
from simpleRegression.utils import printMethodInfo, plot_data
from simpleRegression.fitFunctions import SimpleRegression


def main(regressionMethod, use_interactions, n_features, target_data):

    # Load Norm data
    NormData = pd.read_csv(
        "src/GetDataFromScratch/ScratchData/MaterialSweep/NormData.csv"
    )

    # Extract data
    if n_features == 2:
        features = np.array([NormData["syNorm"], NormData["n"]])
    elif n_features == 3:
        features = np.array([NormData["syNorm"], NormData["n"], NormData["wNorm"]])

    target = NormData[target_data]

    # Perform regression
    if regressionMethod == "linear":
        degree = 1
    elif regressionMethod == "secondOrder":
        degree = 2

    best_params, r2, mse = SimpleRegression(
        features=features,
        target=target,
        degree=degree,
        use_interactions=use_interactions,
    )

    printMethodInfo(regressionMethod, best_params, r2, mse)

    # Plotting
    plot_data(
        features=features,
        target=target,
        method_name=regressionMethod,
        best_params=best_params,
        include_interactions=use_interactions,
    )


if __name__ == "__main__":
    main(
        regressionMethod="secondOrder",
        use_interactions=False,
        n_features=2,
        target_data="fnNorm",
    )
