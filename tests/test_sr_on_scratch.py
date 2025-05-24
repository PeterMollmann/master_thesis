import numpy as np
from SRScratch.SRScratch import SRScratch
from SRScratch.SRScratchUtils.sympyConversion import sympyConversion
import SRScratch.SRScratchUtils.postProcess as pp
from alpine.data import Dataset
from GetDataFromScratch.GetTopographyData import GetTopographyData, LoadScratchTestData

import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def test_sr_on_scratch():
    # LoadScratchTestData()

    NormData = pd.read_csv(
        "src/GetDataFromScratch/ScratchData/MaterialSweep/NewNormData.csv"
    )

    random_seed = 42
    features = np.array([NormData["syNorm"], NormData["n"]])
    target = NormData["H_s/E"]
    X_train, X_test, y_train, y_test = train_test_split(
        features.T, target, test_size=0.2, random_state=random_seed
    )

    yamlPath = "test_sr_on_scratch.yaml"
    yamlfile = yamlPath
    filename = os.path.join(os.path.dirname(__file__), yamlfile)
    with open(filename) as config_file:
        config_file_data = yaml.safe_load(config_file)

    train_data = Dataset("dataset", X_train, y_train)
    test_data = Dataset("dataset", X_test, y_test)
    train_data.X = [train_data.X[:, i] for i in range(X_train.shape[1])]
    test_data.X = [test_data.X[:, i] for i in range(X_train.shape[1])]

    best_ind, fit_score, train_pred, test_pred = SRScratch(
        config_file_data=config_file_data,
        trainDataSet=train_data,
        testDataSet=test_data,
    )

    sC = sympyConversion(tree=best_ind, best_ind_consts=best_ind.consts)
    simple_str = sC.simplify_expr()
    print("Simplified is:")
    print(f"{simple_str}")

    MSE_test = mean_squared_error(y_test, test_pred)
    r2_test = r2_score(y_test, test_pred)

    print("MSE on the test set = ", MSE_test)
    print("R^2 on the test set = ", r2_test)


if __name__ == "__main__":
    test_sr_on_scratch()
