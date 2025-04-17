# loads the .yaml file connected to the problem
# takes a string containing the path to the .yaml file
#
# import by: from main.loadYaml import *
#
# use as follows:
#
# config_file_data = loadYaml("path_name")

import os
import yaml


def loadYaml(path: str):
    yamlfile = path
    filename = os.path.join(os.path.dirname(__file__), yamlfile)
    with open(filename) as config_file:
        config_file_data = yaml.safe_load(config_file)
    return config_file_data
