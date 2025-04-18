import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from simpleRegression.utils import printMethodInfo, plot_data
from simpleRegression.fitFunctions import SimpleRegression


def main():

    # Load Norm data
    NormData = pd.read_csv(
        "src/GetDataFromScratch/ScratchData/MaterialSweep/NormData.csv"
    )

    # Extract data
    features = np.array([NormData["syNorm"], NormData["n"], NormData["wNorm"]])
    # features = np.array([NormData["syNorm"], NormData["n"]])
    target = NormData["fnNorm"]

    # Perform regression
    regressionMethod = "secondOrder"
    use_interactions = True

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
    main()
