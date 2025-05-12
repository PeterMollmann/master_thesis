import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simpleRegression.utils import printMethodInfo, plot_data
from simpleRegression.fitFunctions import fittingFunction, SimpleRegression
from simpleRegression.LiEtAlFunction import LiEtAlFunction, LiEtAlPlot
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def main(
    regressionMethod,
    use_interactions,
    n_features: int = 2,
    target_data: str = "Ft/Fn",
    save_fig: bool = False,
    data_splitting: bool = True,
    featureNorm: bool = True,
):

    # Load raw data
    RawData = pd.read_csv(
        "src/GetDataFromScratch/ScratchData/MaterialSweep/AllRawData.csv"
    )

    # Load Norm data
    NormData = pd.read_csv(
        "src/GetDataFromScratch/ScratchData/MaterialSweep/NewNormData.csv"
        # "src/GetDataFromScratch/ScratchData/MaterialSweep/NormData.csv"
        # "src/GetDataFromScratch/ScratchData/MaterialSweep/ZhangData.csv"
    )

    # Extract data
    if n_features == 2:
        features = np.array([NormData["syNorm"], NormData["n"]])
        xlabel = r"$\sigma_y/E$"
        ylabel = r"$n$"

    zlabel = rf"${target_data}$"
    target = NormData[target_data]

    # Perform regression
    if regressionMethod == "linear":
        degree = 1
    elif regressionMethod == "secondOrder":
        degree = 2

    # Data standardization
    if featureNorm:
        features = (features - np.mean(features, axis=1).reshape(-1, 1)) / np.std(
            features, axis=1
        ).reshape(-1, 1)
        target = (target - np.mean(target)) / np.std(target)

    if data_splitting:
        # Data train test splitting
        random_seed = 42
        X_train, X_test, y_train, y_test = train_test_split(
            features.T, target, test_size=0.2, random_state=random_seed
        )
    else:
        X_train = features.T
        y_train = target

    ##### Using own fitting function #####
    best_params, r2_train, mse_train = SimpleRegression(
        features=X_train.T,
        target=y_train,
        degree=degree,
        use_interactions=use_interactions,
    )
    if data_splitting:
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
        test_r2_score=(
            r2_score(y_true=y_test, y_pred=y_pred) if data_splitting else None
        ),
        test_mse=(
            mean_squared_error(y_true=y_test, y_pred=y_pred) if data_splitting else None
        ),
    )

    # Plotting
    if data_splitting:
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

    ##### Using sklearn #####
    # print(X_train.shape)
    # poly = PolynomialFeatures(
    #     degree=2, include_bias=False, interaction_only=not use_interactions
    # )
    # X_poly = poly.fit_transform(X_train, y_train)
    # model = LinearRegression()
    # model.fit(X_poly, y_train)
    # if data_splitting:
    #     y_poly_pred = model.predict(poly.transform(X_test))
    #     r2 = r2_score(y_test, y_poly_pred)
    #     mse = mean_squared_error(y_test, y_poly_pred)
    #     print(f"R^2: {r2:.4f}, MSE: {mse:.4f}")
    # print("Coefficients:", model.coef_)
    # print("Intercept:", model.intercept_)

    ##### Using LiEtAl function #####

    # Fn_r2_prev = 0
    # Ft_r2_prev = 0

    # for i in np.arange(0.1, 2, 0.05):
    #     for j in np.arange(0.1, 2, 0.05):
    #         for k in np.arange(0.1, 2, 0.05):

    #             Fn, Ft = LiEtAlFunction(
    #                 h_r=RawData["Residual Depth"],
    #                 h_p=RawData["Pile-up Height"],
    #                 w=RawData["Width"] / 2,
    #                 sigma_y=RawData["Yield Strength"],
    #                 n=RawData["Strain Hardening"],
    #                 E=200000,
    #                 mu=0.0,
    #                 alpha=135.6 / 2,
    #                 k_1=i,
    #                 k_2=j,
    #                 k_n=k,
    #             )
    #             Fn_r2 = r2_score(Fn, RawData["Normal Force"])
    #             Ft_r2 = r2_score(Ft, RawData["Tangential Force"])
    #             # if Fn_r2 > Fn_r2_prev and Ft_r2 > Ft_r2_prev:
    #             if Fn_r2 + Ft_r2 > Fn_r2_prev + Ft_r2_prev:
    #                 print(
    #                     f"Fn R2: {Fn_r2:.4f}, Ft R2: {Ft_r2:.4f}, k_1: {i}, k_2: {j}, k_n: {k}"
    #                 )
    #                 k_1_save = i
    #                 k_2_save = j
    #                 k_n_save = k
    #                 Fn_r2_prev = Fn_r2
    #                 Ft_r2_prev = Ft_r2

    # print("Best k_1", k_1_save)
    # print("Best k_2", k_2_save)
    # print("Best k_n", k_n_save)
    # Fn, Ft = LiEtAlFunction(
    #     h_r=RawData["Residual Depth"],
    #     h_p=RawData["Pile-up Height"],
    #     w=RawData["Width"] / 2,
    #     sigma_y=RawData["Yield Strength"],
    #     n=RawData["Strain Hardening"],
    #     E=200000,
    #     mu=0.0,
    #     alpha=135.6 / 2,
    #     k_1=k_1_save,
    #     k_2=k_2_save,
    #     k_n=k_n_save,
    #     #k_1=1.45,
    #     #k_2=1.5,
    #     #k_n=1.15,
    # )
    # print("Best Fn R2", Fn_r2_prev)
    # print("Best Ft R2", Ft_r2_prev)

    # LiEtAlPlot(RawData, Fn, Ft)


if __name__ == "__main__":
    loop = False
    save_fig = False
    data_splitting = True
    featureNorm = True
    use_interactions = True

    if loop:
        regressionMethods = ["linear", "secondOrder"]
        use_interactions = [True, False]
        n_features = [2]
        target_data = ["Ft/Fn", "h_p/h_r", "H_s/E"]
        # target_data = ["fnNorm", "ftNorm", "hrNorm", "hpNorm", "wNorm"]

        for i in regressionMethods:
            for k in n_features:
                for l in target_data:
                    if i == "secondOrder":
                        for j in use_interactions:
                            # print(i, k, l, j)
                            main(
                                regressionMethod=i,
                                use_interactions=j,
                                n_features=k,
                                target_data=l,
                                save_fig=save_fig,
                                data_splitting=data_splitting,
                                featureNorm=featureNorm,
                            )
                    else:
                        # print(i, k, l)
                        main(
                            regressionMethod=i,
                            use_interactions=False,
                            n_features=k,
                            target_data=l,
                            save_fig=save_fig,
                            data_splitting=data_splitting,
                            featureNorm=featureNorm,
                        )
    else:
        main(
            regressionMethod="secondOrder",
            use_interactions=use_interactions,
            n_features=2,
            target_data="h_p/h_r",
            # target_data="Ft/Fn",
            # target_data="H_s/E",
            # target_data="hrNorm",
            save_fig=save_fig,
            data_splitting=data_splitting,
            featureNorm=featureNorm,
        )
