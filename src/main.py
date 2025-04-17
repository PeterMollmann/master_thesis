from simpleRegression.fitFunctions import linearFit, secondOrderFit, secondOrderFit2, secondOrderFit3
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from utils import printMethodInfo


def main():

    # Load Norm data
    NormData = pd.read_csv(
        "src/GetDataFromScratch/ScratchData/MaterialSweep/NormData.csv")
    x = NormData['syNorm']
    y = NormData['n']
    z = NormData['fnNorm']

    X = np.array(x).reshape(40, 5)
    Y = np.array(y).reshape(40, 5)
    Z = np.array(z).reshape(40, 5)

    # Fit the data
    linearFit_popt, _ = curve_fit(linearFit, (x, y), z, p0=(1, 1, 1))
    secondOrderFit_popt, _ = curve_fit(
        secondOrderFit, (x, y), z, p0=(1, 1, 1, 1, 1))
    secondOrderFit2_popt, _ = curve_fit(
        secondOrderFit2, (x, y), z, p0=(1, 1, 1, 1))
    secondOrderFit3_popt, _ = curve_fit(
        secondOrderFit3, (x, y), z, p0=(1, 1, 1, 1, 1, 1))

    # Print the optimal parameters and r2 score
    printMethodInfo("Linear Fit", linearFit_popt,
                    r2_score(z, linearFit((x, y), *linearFit_popt)))
    printMethodInfo("Second Order Fit", secondOrderFit_popt,
                    r2_score(z, secondOrderFit((x, y), *secondOrderFit_popt)))
    printMethodInfo("Second Order Fit 2", secondOrderFit2_popt,
                    r2_score(z, secondOrderFit2((x, y), *secondOrderFit2_popt)))
    printMethodInfo("Second Order Fit 3", secondOrderFit3_popt,
                    r2_score(z, secondOrderFit3((x, y), *secondOrderFit3_popt)))

    # plot data and fit
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(X, Y, Z, cmap='viridis', label='Data')
    ax.plot_surface(X, Y, linearFit((X, Y), *linearFit_popt), color='r',
                    alpha=0.5, label='Linear Fit')
    ax.plot_surface(X, Y, secondOrderFit((X, Y), *secondOrderFit_popt),
                    color='b', alpha=0.5, label='Second Order Fit')
    ax.plot_surface(X, Y, secondOrderFit2((X, Y), *secondOrderFit2_popt),
                    color='g', alpha=0.5, label='Second Order Fit 2')
    ax.plot_surface(X, Y, secondOrderFit3((X, Y), *secondOrderFit3_popt),
                    color='y', alpha=0.5, label='Second Order Fit 3')

    ax.scatter(x, y, z, color='k', label='Data Points')
    ax.set_xlabel('syNorm')
    ax.set_ylabel('n')
    ax.set_zlabel('fnNorm')
    ax.set_title('3D Surface Plot of Data')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
