import numpy as np
from SRScratch.SRScratch import SRScratch
import os
import yaml


def test_simple_sr():

    x = np.array([x/10. for x in range(-10, 10)])
    y = 2*x**3 + 1.2*x**2 + 4.1*x

    yamlPath = "test_simple_sr.yaml"
    yamlfile = yamlPath
    filename = os.path.join(os.path.dirname(__file__), yamlfile)
    with open(filename) as config_file:
        config_file_data = yaml.safe_load(config_file)

    best_ind = SRScratch(config_file_data=config_file_data,
                         features=x, target=y)
    print(best_ind)


if __name__ == "__main__":
    test_simple_sr()
