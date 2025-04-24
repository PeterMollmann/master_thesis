import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simpleRegression.utils import printMethodInfo, plot_data
from simpleRegression.fitFunctions import fittingFunction, SimpleRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def main(regressionMethod, use_interactions, n_features, target_data, save_fig=False):

    # Load Norm data
    NormData = pd.read_csv(
        "src/GetDataFromScratch/ScratchData/MaterialSweep/NormData.csv"
    )

    # Extract data
    if n_features == 2:
        features = np.array([NormData["syNorm"], NormData["n"]])
        xlabel = r"$\sigma_y/E$"
        ylabel = r"$n$"
    elif n_features == 3:
        features = np.array([NormData["syNorm"], NormData["n"], NormData["wNorm"]])
        xlabel = r"$\sigma_y/E$"
        ylabel = r"$n$"

    if target_data == "fnNorm":
        zlabel = r"$f_n/Ew^2$"
    elif target_data == "ftNorm":
        zlabel = r"$f_t/Eh_d\sqrt{wh_d}$"
    elif target_data == "hrNorm":
        zlabel = r"$h_r/h_d$"
    elif target_data == "hpNorm":
        zlabel = r"$h_p/h_d$"
    target = NormData[target_data]

    # Perform regression
    if regressionMethod == "linear":
        degree = 1
    elif regressionMethod == "secondOrder":
        degree = 2

    # Data train test splitting
    X_train, X_test, y_train, y_test = train_test_split(
        features.T, target, test_size=0.2, random_state=42
    )

    best_params, r2_train, mse_train = SimpleRegression(
        features=X_train.T,
        target=y_train,
        degree=degree,
        use_interactions=use_interactions,
    )

    # Fitted function on test data
    y_pred = fittingFunction(
        X_test.T,
        *best_params,
        degree=degree,
        include_interactions=use_interactions,
    )

    printMethodInfo(
        method_name=regressionMethod,
        target_name=target_data,
        best_params=best_params,
        n_features=n_features,
        use_interactions=use_interactions,
        train_r2_score=r2_train,
        train_mse=mse_train,
        test_r2_score=r2_score(y_true=y_test, y_pred=y_pred),
        test_mse=mean_squared_error(y_true=y_test, y_pred=y_pred),
    )

    # Plotting
    plot_data(
        features_all=features,
        target_all=target,
        features_test=X_test.T,
        target_test=y_test,
        method_name=regressionMethod,
        best_params=best_params,
        include_interactions=use_interactions,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        target_name=target_data,
        save_fig=save_fig,
    )


if __name__ == "__main__":
    loop = False

    if loop:
        regressionMethods = ["linear", "secondOrder"]
        use_interactions = [True, False]
        n_features = [2, 3]
        target_data = ["fnNorm", "ftNorm", "hrNorm", "hpNorm"]

        for i in regressionMethods:
            for k in n_features:
                for l in target_data:
                    if i == "secondOrder":
                        for j in use_interactions:
                            print(i, k, l, j)
                            main(
                                regressionMethod=i,
                                use_interactions=j,
                                n_features=k,
                                target_data=l,
                                save_fig=True,
                            )
                    else:
                        print(i, k, l)
                        main(
                            regressionMethod=i,
                            use_interactions=False,
                            n_features=k,
                            target_data=l,
                            save_fig=True,
                        )
    else:
        main(
            regressionMethod="secondOrder",
            use_interactions=False,
            n_features=2,
            target_data="hrNorm",
            save_fig=False,
        )
