import numpy as np
from SRScratch.SRScratch import SRScratch
from alpine.data import Dataset

import os
import yaml


def test_simple_sr():

    x = np.array([x/10. for x in range(-10, 10)])
    y = x**3 + x**2 + x

    yamlPath = "test_simple_sr.yaml"
    yamlfile = yamlPath
    filename = os.path.join(os.path.dirname(__file__), yamlfile)
    with open(filename) as config_file:
        config_file_data = yaml.safe_load(config_file)

    train_data = Dataset("dataset", x, y)
    train_data.X = [train_data.X]

    seed = [
        "add(add(add(mul(mul(x, mul(x, x)),x), mul(x,mul(x, x))), mul(x, x)), x)"]

    _, fit_score = SRScratch(config_file_data=config_file_data,
                             trainDataSet=train_data, seed=seed)
    assert fit_score <= 1e-12


if __name__ == "__main__":
    test_simple_sr()
