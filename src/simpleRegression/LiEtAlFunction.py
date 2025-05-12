import numpy as np
import matplotlib.pyplot as plt


def LiEtAlFunction(
    h_r,
    h_p,
    w,
    sigma_y: float = 500,
    n: float = 0.5,
    E: int = 200000,
    mu: float = 0.0,
    alpha: float = 135.6 / 2,
    k_1: float = 1.7,
    k_2: float = 2.0,
    k_n: float = 1.35,
):
    """
    Li et al. 2025 proposed a model to calculate the normal and tangential forces.

    params:
        h_r: Residual depth
        h_p: Pile-up height
        w: Width
        sigma_y: Yield strength
        n: Strain hardening exponent
        E: Young's modulus
        mu: Friction coefficient
        alpha: half angle of edge to edge
        k_1: constant
        k_2: constant
        k_n: constant
    returns:
        Fn: Normal force
        Ft: Tangential force
    """

    alpha = alpha * np.pi / 180  # Convert to radians
    h_rp = h_r + h_p
    V_p = 2 / 3 * (w - 3.5 * h_r) * h_p * 4.95 * h_r
    V = 4.09 * h_rp**3 - V_p
    c = (V / (np.pi * sigma_y / E)) ** (1 / 3)

    Fn = (
        (np.pi * sigma_y * (k_1 * c) ** (3 * n))
        / ((2 - 3 * n) * np.sin(alpha))
        * ((k_n * w) ** (2 - 3 * n) - (k_2 * h_rp) ** (2 - 3 * n))
    )

    Ft = (k_1**2 * 2 * alpha) / (E) * (
        (sigma_y**2 * c**2 * (n - 1))
        / (4 * (n + 1))
        * (1 - ((k_2 * h_rp) / (k_1 * c)) ** 2)
        + (sigma_y**2 * c**2)
        / ((n + 1) * (3 * n + 1))
        * (((k_1 * c) / (k_2 * h_rp)) ** (3 * n + 1) - 1)
    ) + mu * Fn

    return abs(Fn), Ft


def LiEtAlPlot(RawData, Fn, Ft):
    plt.subplot(1, 2, 1)
    plt.scatter(RawData["Yield Strength"], RawData["Normal Force"], label="Fn Data")
    plt.scatter(
        RawData["Yield Strength"],
        Fn,
        label="Fn LiEtAl",
        marker="x",
        color="red",
        s=100,
    )
    plt.xlabel("Yield Strength")
    plt.ylabel("Force")
    plt.title("LiEtAl Function")
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.scatter(RawData["Yield Strength"], RawData["Tangential Force"], label="Ft Data")
    plt.scatter(
        RawData["Yield Strength"],
        Ft,
        label="Ft LiEtAl",
        marker="x",
        color="green",
        s=100,
    )
    plt.xlabel("Yield Strength")
    plt.ylabel("Force")
    plt.title("LiEtAl Function")
    plt.legend()
    plt.grid()
    plt.show()
