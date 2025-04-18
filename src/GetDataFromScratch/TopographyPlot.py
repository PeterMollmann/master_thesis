import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# function that plots the topography of the scratch
# the z coordinates are plotted along the x direction
# the x coordinates are plotted along the y direction
# the y coordinates are plotted along the z direction


def TopographyPlot(z, x, y, title, modelSize="Full"):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    if modelSize == "Full":
        # Extend data to full domain
        z = np.append(z, z)
        y = np.append(y, y)
        x = np.append(x, -x)
        ax.set_ylim([-0.32, 0.32])
    else:
        # Only showing half the domain
        ax.set_ylim([0, 0.32])

        # Plotting
    surf = ax.plot_trisurf(z, x, y, cmap=cm.coolwarm, edgecolor="none")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # contour = ax.contour(z, x, y, zdir='z', offset=ax.get_zlim()[
    #                      0], cmap=cm.coolwarm)
    # ax.clabel(contour, fontsize=8, colors='k')
    ax.set_zlim([0.55, 0.65])
    ax.set_title(title)
    ax.set_xlim([0, 2.88])
