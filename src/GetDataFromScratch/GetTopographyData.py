import numpy as np


def GetTopographyData(x: np.ndarray, y: np.ndarray, z: np.ndarray, lowerBound: float, upperBound: float):
    """ Gets the residual scratch depth, the scratch width and the pile-up height.

    This function takes the x (depth direction), y (indentation direction), and z (scratch direction) coordinates of a scratch test 
    and returns residual depth, scratch width and pile-up height. 

    Args:
        x (np.ndarray): x-coordinates (depth direction).
        y (np.ndarray): y-coordinates (indentation direction).
        z (np.ndarray): z-coordinates (scratch direction).
        lowerBound (float): Lower bound for extracting specific set of coordinates along z-direction (scratch direction).
        upperBound (float): Upper bound for extracting specific set of coordinates along z-direction (scratch direction).

    Returns:
        list: residual scratch depth, scratch width and pile-up height

    """

    # create mask for selecting specific set of data
    mask = (z >= lowerBound) & (z <= upperBound)
    xMask = x[mask]
    yMask = y[mask]

    # Getting scratch depth normalised wrt. domain size (0.64mm)
    residualSratchDepth = np.abs(np.min(yMask) - 0.64)

    # Getting pile-up height normalised wrt. domain size (0.64mm)
    pileUpHeight = np.max(yMask) - 0.64

    # Getting scratch width
    idxOfPileUpheight = np.argmax(yMask)
    scratchWidth = 2*xMask[idxOfPileUpheight]

    return residualSratchDepth, scratchWidth, pileUpHeight
