from alpine.gp import gpsymbreg as gps
from alpine.data import Dataset
from SRScratch.SRScratchUtils.gpsrHelpers import *
from SRScratch.SRScratchUtils.loadYaml import loadYaml
import numpy as np
import os
import yaml


def SRScratch(config_file_data, trainDataSet, testDataSet, seed=None):
    """
    Args:
        yamlPath (str): Path to .yaml config file for the problem
        features (array-like): Array with features data. Different features as columns.
        target (array-like): Array with target data.
        seed (str): Seed for the initial population.

    Returns:

    """
    # load config settings

    # config_file_data = loadYaml(path=yamlPath)

    # Initialise primitive set
    pset = gp.PrimitiveSetTyped("Main", [float, float], float)
    pset.renameArguments(ARG0="x1")
    pset.renameArguments(ARG1="x2")

    pset.addTerminal(object, float, "a")

    # def protectedDiv(left, right):
    #     try:
    #         return left / right
    #     except ZeroDivisionError:
    #         return np.nan

    # pset.addPrimitive(protectedDiv, [float, float], float)

    batch_size = config_file_data["gp"]["batch_size"]
    callback_func = assign_attributes

    fitness_scale = config_file_data["gp"]["fitness_scale"]
    penalty = config_file_data["gp"]["penalty"]
    common_data = {"penalty": penalty, "fitness_scale": fitness_scale}

    gpsr = gps.GPSymbolicRegressor(
        pset=pset,
        fitness=compute_attributes.remote,
        predict_func=eval_expr.remote,
        error_metric=compute_MSEs.remote,
        common_data=common_data,
        callback_func=callback_func,
        print_log=True,
        num_best_inds_str=1,
        config_file_data=config_file_data,
        save_best_individual=False,
        output_path="./",
        seed=seed,
        plot_best_individual_tree=0,
        batch_size=batch_size,
    )
    gpsr.fit(train_data=trainDataSet)

    best_ind = gpsr.best
    # print(best_ind)
    fit_score = gpsr.score(trainDataSet)
    train_pred = gpsr.predict(trainDataSet)
    test_pred = gpsr.predict(testDataSet)

    ray.shutdown()
    # print(fit_score)

    return best_ind, fit_score, train_pred, test_pred
