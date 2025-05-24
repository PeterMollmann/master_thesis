import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib import rcParams


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


def LiEtAlToNonDimGroups(Fn, Ft, w, E):
    """
    Convert the forces to non-dimensional groups.
    params:
        Fn: Normal force
        Ft: Tangential force
        w: Width
        E: Young's modulus
    returns:
        H_sE: Scratch hardness non-dimensional group
        FtFn: Friction non-dimensional group
    """

    Fn = Fn.reshape(40, 5)
    Ft = Ft.reshape(40, 5)
    w = w.reshape(40, 5)

    H_s = Fn / (1 / 4 * w**2)
    H_sE = H_s / E
    FtFn = Ft / Fn

    return H_sE, FtFn


def LiEtAlPlot(Sy, n, LiEtAlForce, TargetForce, ylabel=""):

    rcParams.update(
        {
            "font.family": "serif",  # or "DejaVu Serif"
            "mathtext.fontset": "cm",  # Use Computer Modern (LaTeX default)
            "mathtext.rm": "serif",
            "font.size": 12,
        }
    )

    Sy = Sy.reshape(40, 5)
    n = n.reshape(40, 5)
    LiEtAlForce = LiEtAlForce.reshape(40, 5)
    TargetForce = TargetForce.reshape(40, 5)

    colors = ["m", "r", "g", "b", "k"]
    shapes = ["*", "o", "x", "^", "+"]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    for i in range(len(n[1, :])):
        plt.scatter(
            Sy[:, i],
            TargetForce[:, i],
            color=colors[i],
            marker=shapes[i],
            label=rf"$n={n[1,i]}$",
        )
        plt.plot(
            Sy.reshape(40, 5)[:, i],
            LiEtAlForce.reshape(40, 5)[:, i],
            color=colors[i],
            linestyle="--",
            label="Li et al.",
        )

    if ylabel == "Fn":
        plt.ylim([0, 300])
        plt.ylabel(r"$F_n$ [N]")
    elif ylabel == "Ft":
        plt.ylim([0, 70])
        plt.ylabel(r"$F_t$ [N]")
    plt.xlim([0, 2000])

    plt.xlabel(r"$\sigma_y [MPa]$")

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=5,
        fancybox=True,
        shadow=True,
        fontsize="small",
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("STIXGeneral")
    ax.set_axisbelow(True)

    plt.grid()
    plt.show()
