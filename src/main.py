import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simpleRegression.utils import printMethodInfo, plot_data
from simpleRegression.fitFunctions import fittingFunction, SimpleRegression
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
):

    # Load Norm data
    NormData = pd.read_csv(
        "src/GetDataFromScratch/ScratchData/MaterialSweep/NewNormData.csv"
    )

    # Extract data
    if n_features == 2:
        features = np.array([NormData["syNorm"], NormData["n"]])
        xlabel = r"$\sigma_y/E$"
        ylabel = r"$n$"

    zlabel = rf"${{{target_data}}}$"
    target = NormData[target_data]

    # Perform regression
    if regressionMethod == "linear":
        degree = 1
    elif regressionMethod == "secondOrder":
        degree = 2

    # Data standardization
    features = (features - np.mean(features, axis=1).reshape(-1, 1)) / np.std(
        features, axis=1
    ).reshape(-1, 1)
    target = (target - np.mean(target)) / np.std(target)

    # Data train test splitting
    random_seed = 42
    X_train, X_test, y_train, y_test = train_test_split(
        features.T, target, test_size=0.2, random_state=random_seed
    )

    best_params, r2_train, mse_train = SimpleRegression(
        features=X_train.T,
        target=y_train,
        degree=degree,
        use_interactions=use_interactions,
    )

    ##### Using own fitting function #####
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

    ##### Using sklearn #####
    # print(X_train.shape)
    poly = PolynomialFeatures(
        degree=2, include_bias=False, interaction_only=not use_interactions
    )
    X_poly = poly.fit_transform(X_train, y_train)
    model = LinearRegression()
    model.fit(X_poly, y_train)
    y_poly_pred = model.predict(poly.transform(X_test))
    r2 = r2_score(y_test, y_poly_pred)
    mse = mean_squared_error(y_test, y_poly_pred)
    print(f"R^2: {r2:.4f}, MSE: {mse:.4f}")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)


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
                            # print(i, k, l, j)
                            main(
                                regressionMethod=i,
                                use_interactions=j,
                                n_features=k,
                                target_data=l,
                                save_fig=True,
                            )
                    else:
                        # print(i, k, l)
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
            use_interactions=True,
            n_features=2,
            target_data="H_s/E",
            save_fig=False,
        )
