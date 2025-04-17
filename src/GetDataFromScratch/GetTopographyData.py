import numpy as np
import glob


def LoadScratchTestData(fileNameID: str, path: str, toLoad: str = "both"):
    """ Loads the coodinates and reaction forces from a scratch test.

    This function takes the file name and what data to load and returns the data.

    Args:
        fileNameID (str): The ID of the data file. Does not include "reactionForces" or "coordinates"
        path (str): The path to the data files. 
        toLoad (str): What data to load. "coords", "RFs", "both"

    Returns:
        np.ndarray: The data of either the coordinates, reaction forces or both 

    """
    if toLoad == "RFs":
        rfs = np.loadtxt(path + "reactionForces_"+fileNameID+".txt", delimiter=",",
                         skiprows=2, usecols=(1, 2, 3, 4))
        return rfs

    elif toLoad == "coords":
        coords = np.loadtxt(path + "coordinates_"+fileNameID+".txt",
                            delimiter=",", skiprows=1, usecols=(1, 2, 3, 4, 5, 6))
        return coords

    else:
        coords = np.loadtxt(path + "coordinates_"+fileNameID+".txt",
                            delimiter=",", skiprows=1, usecols=(1, 2, 3, 4, 5, 6))
        rfs = np.loadtxt(path + "reactionForces_"+fileNameID+".txt", delimiter=",",
                         skiprows=2, usecols=(1, 2, 3, 4))
        return rfs, coords


def GetTopographyData(coords: np.ndarray, lowerBound: float = 2.00, upperBound: float = 2.44):
    """ Gets the residual scratch depth, the scratch width and the pile-up height.

    This function takes undeformed and deformed coordinates of a scratch test 
    and returns residual depth, scratch width and pile-up height. 

    Only works if the height of the substrate is 0.64mm, otherwise, change the function.

    Args:
        coords (array-like): Coordinates of both undeformed and deformed nodes.
        lowerBound (float): Lower bound for extracting specific set of coordinates along z-direction (scratch direction).
        upperBound (float): Upper bound for extracting specific set of coordinates along z-direction (scratch direction).

    Returns:
        list: residual scratch depth, scratch width and pile-up height

    """
    x_undef, y_undef, z_undef = coords[:, 0], coords[:, 1], coords[:, 2]
    x_def, y_def, z_def = coords[:, 3], coords[:, 4], coords[:, 5]

    # create mask for selecting specific set of data
    z_direction_mask = (z_undef >= lowerBound) & (z_undef <= upperBound)

    x_undef_masked = x_undef[z_direction_mask]
    x_def_masked = x_def[z_direction_mask]
    y_def_masked = y_def[z_direction_mask]

    x_undef_unique = np.unique(x_undef_masked)
    residualSratchDepth = 1.0
    pileUpHeight = 0.0
    for unique_value in x_undef_unique:
        x_direction_mask = (x_undef_masked == unique_value)
        temp = np.mean(y_def_masked[x_direction_mask])
        if temp < residualSratchDepth:
            residualSratchDepth = temp
        elif temp > pileUpHeight:
            pileUpHeight = temp
            xUniqueOfMaxPileUp = x_direction_mask

    scratchWidth = 2*np.mean(x_def_masked[xUniqueOfMaxPileUp])

    return abs(residualSratchDepth-0.64), scratchWidth, pileUpHeight-0.64


def GetData(dataFolderPath):

    for name in glob.glob(dataFolderPath+"coordinates_*"):
        coords = LoadScratchTestData(name, )

    pass


def GetScratchProfile(coords: np.ndarray, lowerBound: float = 2.00, upperBound: float = 2.44):
    """ Gets the average scratch profile between the lower and upper bound normal to the x-y plane. 

    This function takes undeformed and deformed coordinates of a scratch test 
    and returns the average scratch profile coordinates in the x-y plane. 

    Only works if the height of the substrate is 0.64mm, otherwise, change the function.

    Args:
        coords (array-like): Coordinates of both undeformed and deformed nodes.
        lowerBound (float): Lower bound for extracting specific set of coordinates along z-direction (scratch direction).
        upperBound (float): Upper bound for extracting specific set of coordinates along z-direction (scratch direction).

    Returns:
        list: average profile coordinate x, average profile coordinate y.

    """
    x_undef, y_undef, z_undef = coords[:, 0], coords[:, 1], coords[:, 2]
    x_def, y_def, z_def = coords[:, 3], coords[:, 4], coords[:, 5]
    z_direction_mask = (z_undef >= lowerBound) & (z_undef <= upperBound)

    x_undef_masked = x_undef[z_direction_mask]
    x_def_masked = x_def[z_direction_mask]
    y_def_masked = y_def[z_direction_mask]

    x_undef_unique = np.unique(x_undef_masked)
    x_def_unique_mean = np.zeros(shape=len(x_undef_unique))
    y_def_unique_mean = np.zeros(shape=len(x_undef_unique))
    for i, unique_value in enumerate(x_undef_unique):
        x_direction_mask = (x_undef_masked == unique_value)
        x_def_unique_mean[i] = np.mean(x_def_masked[x_direction_mask])
        y_def_unique_mean[i] = np.mean(y_def_masked[x_direction_mask])

    return x_def_unique_mean, y_def_unique_mean-0.64
