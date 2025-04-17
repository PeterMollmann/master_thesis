import numpy as np


def linearFit(xy, a, b, c):
    """ A linear fitting function for 3D data with the form z = a + b*x + c*y.
    Takes the x and y data in an array xy, and the parameters a, b, c.

    Args:
        xy (array-like): A tuple containing the x and y data.
        a (float): The intercept.
        b (float): The coefficient for x.
        c (float): The coefficient for y.
    Returns:
        array-like: The fitted values.
    """
    x, y = xy
    return a + b*x + c*y


def secondOrderFit(xy, a, b, c, d, e):
    """ A second order fitting function for 3D data with the form z = a + b*x + c*y + d*x^2 + e*y^2.
    Takes the x and y data in an array xy, and the parameters a, b, c, d, e.

    Args:
        xy (array-like): A tuple containing the x and y data.
        a (float): The intercept.
        b (float): The coefficient for x.
        c (float): The coefficient for y.
        d (float): The coefficient for x^2.
        e (float): The coefficient for y^2.
    Returns:
        array-like: The fitted values.
    """
    x, y = xy
    return a + b*x + c*y + d*x**2 + e*y**2


def secondOrderFit2(xy, a, b, c, d):
    """ A second order fitting function for 3D data with the form z = a + b*x + c*y + d*x*y.
    Takes the x and y data in an array xy, and the parameters a, b, c, d.

    Args:
        xy (array-like): A tuple containing the x and y data.
        a (float): The intercept.
        b (float): The coefficient for x.
        c (float): The coefficient for y.
        d (float): The coefficient for xy.
    Returns:
        array-like: The fitted values.
    """
    x, y = xy
    return a + b*x + c*y + d*x*y


def secondOrderFit3(xy, a, b, c, d, e, f):
    """ A second order fitting function for 3D data with the form z = a + b*x + c*y + d*x^2 + e*y^2 + d*x*y.
    Takes the x and y data in an array xy, and the parameters a, b, c, d, e, f.

    Args:
        xy (array-like): A tuple containing the x and y data.
        a (float): The intercept.
        b (float): The coefficient for x.
        c (float): The coefficient for y.
        d (float): The coefficient for x^2.
        e (float): The coefficient for y^2.
        f (float): The coefficient for xy.
    Returns:
        array-like: The fitted values.
    """
    x, y = xy
    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y


def thirdOrderFit(xy, a, b, c, d, e, f, g):
    """ A third order fitting function for 3D data with the form z = a + b*x + c*y + d*x^2 + e*y^2 + f*x*y + g*x^3.
    Takes the x and y data in an array xy, and the parameters a, b, c, d, e, f, g.

    Args:
        xy (array-like): A tuple containing the x and y data.
        a (float): The intercept.
        b (float): The coefficient for x.
        c (float): The coefficient for y.
        d (float): The coefficient for x^2.
        e (float): The coefficient for y^2.
        f (float): The coefficient for xy.
        g (float): The coefficient for x^3.
    Returns:
        array-like: The fitted values.
    """
    x, y = xy
    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y + g*x**3
