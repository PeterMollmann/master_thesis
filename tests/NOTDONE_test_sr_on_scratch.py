import numpy as np
from SRScratch.SRScratch import SRScratch
from alpine.data import Dataset
from GetDataFromScratch.GetTopographyData import GetTopographyData, LoadScratchTestData

import os
import yaml


def test_sr_on_scratch():
    # LoadScratchTestData()

    yamlPath = "test_simple_sr.yaml"
    yamlfile = yamlPath
    filename = os.path.join(os.path.dirname(__file__), yamlfile)
    with open(filename) as config_file:
        config_file_data = yaml.safe_load(config_file)

    train_data = Dataset("dataset", x, y)
    train_data.X = [train_data.X]

    best_ind, fit_score = SRScratch(
        config_file_data=config_file_data, trainDataSet=train_data
    )
    assert fit_score <= 1e-12


if __name__ == "__main__":
    test_sr_on_scratch()
