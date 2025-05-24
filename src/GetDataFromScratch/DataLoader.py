import numpy as np


def DataLoader(fileNameID, path, toLoad="both"):
    """ Loads the coodinates and reaction forces from a scratch test.

    This function takes the file name and what data to load and returns the data.

    Args:
        fileNameID (str): The ID of the data file. Does not include "reactionForces" or "coordinates"
        path (str): The path to the data files
        toLoad (str): What data to load. "coords", "RFs", "both"

    Returns:
        np.ndarray: The data of either the coordinates, reaction forces or both 

    """
    if toLoad == "RFs":
        rfs = np.loadtxt(path + "reactionForces_"+fileNameID+".txt", delimiter=",",
                         skiprows=2, usecols=(1, 2, 3, 4))
        return rfs
    elif toLoad == "coords":
        # print("ScratchData/coordinates_"+fileNameID+".txt")
        coords = np.loadtxt(path + "coordinates_"+fileNameID+".txt",
                            delimiter=",", skiprows=1, usecols=(4, 5, 6))
        return coords
    else:
        coords = np.loadtxt(path + "coordinates_"+fileNameID+".txt",
                            delimiter=",", skiprows=1, usecols=(4, 5, 6))
        rfs = np.loadtxt(path + "reactionForces_"+fileNameID+".txt", delimiter=",",
                         skiprows=2, usecols=(1, 2, 3, 4))
        return rfs, coords
