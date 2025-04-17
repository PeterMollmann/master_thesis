import numpy as np
from simpleRegression.fitFunctions import linearFit
from scipy.optimize import curve_fit


def test_linear_regression():
    # Example data
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 5, 7, 11])
    z = 2 + 3*x + 4*y
    # Fit the data
    popt, pcov = curve_fit(linearFit, (x, y), z, p0=(1, 1, 1))
    assert np.allclose(
        popt, [2, 3, 4], atol=0.1), f"Expected parameters close to [2, 3, 4], got {popt}"
