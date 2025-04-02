
# %%
import numpy as np
from GetDataFromScratch.GetTopographyData import LoadScratchTestData, GetTopographyData
from GetDataFromScratch.TopographyPlot import TopographyPlot
import matplotlib.pyplot as plt
np.set_printoptions(legacy='1.25')
# %%

yield_strengths = np.arange(50, 2050, 50)
strain_hardening_indexs = np.arange(0.1, 0.6, 0.1)
youngs_modulus = 200000

color = ["b", "r", "g", "y", "k"]

h_rs = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
ws = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
h_ps = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
rf2 = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
rf3 = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))


plt.figure(1)
plt.figure(2)
plt.figure(3)
plt.figure(4)
plt.figure(5)
for i, sy in enumerate(yield_strengths):
    count = 0
    for j, n in enumerate(strain_hardening_indexs):
        # print(yield_strengths)
        fileNameID = "SY"+str(sy)+"_n0"+str(int(10*n)) + \
            "_d0050_E200000_mu00_rho78_Poisson03"
        path = "ScratchData/MaterialSweep/"
        rfs, coords = LoadScratchTestData(
            fileNameID=fileNameID, path=path, toLoad="both")
        # x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        h_r, w, h_p = GetTopographyData(coords, 1.5, 2.0)
        h_rs[i, j] = h_r
        ws[i, j] = w
        h_ps[i, j] = h_p
        rf2[i, j] = rfs[-4, 2]
        rf3[i, j] = rfs[-4, 3]


plt.figure(1)
plt.plot(yield_strengths, ws / (abs(rf2)/youngs_modulus)**0.5)
plt.vlines([600, 600], 0, 60, "k", "--")
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4",
           "n=0.5", r"$\sigma_y=600$ [MPa]"])
plt.xlabel("Yield strength [MPa]")
plt.ylabel(r"$w/(F_n/E)^{0.5}$ [-]")

plt.figure(2)
plt.plot(yield_strengths, ws / (abs(rf2)/yield_strengths.reshape(-1, 1))**0.5)
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel("Yield strength [MPa]")
plt.ylabel(r"$w/(F_n/\sigma_y)^{0.5}$ [-]")


plt.figure(3)
plt.plot(yield_strengths, abs(rf2) / (youngs_modulus*h_rs**2))
plt.vlines([600, 600], 0, 1, "k", "--")
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4",
           "n=0.5", r"$\sigma_y=600$ [MPa]"])
plt.xlabel("Yield strength [MPa]")
plt.ylabel(r"$F_n/(Eh_r^2)$ [-]")


plt.figure(4)
plt.plot(yield_strengths, abs(rf2) /
         ((youngs_modulus*yield_strengths.reshape(-1, 1))**0.5*h_rs**2))
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel("Yield strength [MPa]")
plt.ylabel(r"$F_n/(E\sigma_y)^{0.5}h_r^2$ [-]")


# %%
fileNameID = "SY"+str(250)+"_n0"+str(int(10*0.1)) + \
    "_d0050_E200000_mu00_rho78_Poisson03"
path = "ScratchData/MaterialSweep/"
rfs, coords = LoadScratchTestData(
    fileNameID=fileNameID, path=path, toLoad="both")
x, y, z = coords[:, 3], coords[:, 4], coords[:, 5]
TopographyPlot(z, x, y, title="", modelSize="half")

print(np.unique(coords[:, 0]))
