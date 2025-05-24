import numpy as np
from SRScratch.SRScratch import SRScratch
from alpine.data import Dataset
from sklearn.model_selection import train_test_split

import os
import yaml


def test_simple_sr():

    x1 = np.array([x / 10.0 for x in range(-10, 10)])
    x2 = np.array([x / 2.0 for x in range(-10, 10)])
    x = np.array([x1, x2]).T
    y = x1**4 + x2**3 + x1**2 + x2
    # print(x.shape)

    yamlPath = "test_simple_sr.yaml"
    yamlfile = yamlPath
    filename = os.path.join(os.path.dirname(__file__), yamlfile)
    with open(filename) as config_file:
        config_file_data = yaml.safe_load(config_file)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    train_data = Dataset("dataset", X_train, y_train)
    test_data = Dataset("dataset", X_test, y_test)
    train_data.X = [train_data.X[:, i] for i in range(X_train.shape[1])]
    test_data.X = [test_data.X[:, i] for i in range(X_test.shape[1])]

    # seed = ["add(add(add(mul(mul(x, mul(x, x)),x), mul(x,mul(x, x))), mul(x, x)), x)"]
    seed = [
        "add(add(add(mul(x1,mul(x1,mul(x1,x1))),mul(x2,mul(x2,x2))),mul(x1,x1)),x2)"
    ]

    best_ind, fit_score, train_pred, test_pred = SRScratch(
        config_file_data=config_file_data,
        trainDataSet=train_data,
        testDataSet=test_data,
        seed=seed,
    )
    assert fit_score <= 1e-12


if __name__ == "__main__":
    test_simple_sr()
