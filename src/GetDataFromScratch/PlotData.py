
# %%
import numpy as np
from GetDataFromScratch.GetTopographyData import LoadScratchTestData, GetTopographyData, GetScratchProfile
from GetDataFromScratch.TopographyPlot import TopographyPlot
import matplotlib.pyplot as plt
from matplotlib import cm
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

H_s = abs(rf2)/(1/4*(ws)**2)


plt.figure(1)
plt.plot(yield_strengths/youngs_modulus, ws / (abs(rf2)/youngs_modulus)**0.5)
plt.vlines([600/200000, 600/200000], 0, 60, "k", "--")
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4",
           "n=0.5", r"$\sigma_y=600$ [MPa]"])
plt.xlabel("Yield strength / Youngs modulus [-]")
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

plt.figure(5)
plt.plot(strain_hardening_indexs, rf3[1:-1:5, :].T/abs(rf2[1:-1:5, :].T))
plt.legend(yield_strengths[1:-1:5])
plt.xlabel(r"n [-]")
plt.ylabel(r"$F_t/F_n$ [-]")
plt.ylim([0, 0.4])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
E_star = ((1-0.3**2)/youngs_modulus + (1-0.2**2)/1220000)**(-1)
plt.plot(yield_strengths/(E_star), h_ps/h_rs)
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel(r"$\sigma_y/E$ [N]")
ax.set_xscale("log")
plt.ylabel(r"$h_p/h_r$ [-]")
plt.ylim([0, 1])

fig = plt.figure()
plt.plot(ws, h_ps/h_rs)
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel(r"$w$ [mm]")
# ax.set_xscale("log")
plt.ylabel(r"$h_p$ [mm]")
# plt.ylim([0, 1])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(yield_strengths/youngs_modulus, H_s/yield_strengths.reshape(-1, 1))
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel(r"$\sigma_y/E$ [-]")
ax.set_xscale("log")
ax.set_yscale("log")
plt.ylabel(r"$H_s/\sigma_y$ [-]")
plt.ylim([0, 200])
plt.xlim([0.0001, 0.01])


fnNorm = abs(rf2)/((youngs_modulus*yield_strengths.reshape(-1, 1))**0.5*h_rs**2)
SY, N = np.meshgrid(strain_hardening_indexs, yield_strengths)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(N, SY,  fnNorm, cmap=cm.coolwarm)
ax.set_ylabel("n [-]")
ax.set_xlabel(r"$\sigma_y [MPa]$")
ax.set_zlabel(r"$F_n/((E\sigma_y)^{0.5}h_r^2) [-]$", rotation=90)
ax.set_xlim([2000, 0])
ax.set_ylim([0.1, 0.5])
ax.set_zlim([0, 11])
# fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)

wNorm = ws / (abs(rf2)/yield_strengths.reshape(-1, 1))**0.5
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(SY, N,  wNorm, cmap=cm.coolwarm)
ax.set_xlim([0.5, 0.1])
ax.set_ylim([0, 2000])
ax.set_zlim([0, 1.5])
fig.colorbar(surf, shrink=0.5, aspect=5)

fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
# fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
plt.subplots_adjust(left=None, bottom=None, right=None,
                    top=None, wspace=None, hspace=None)
surf = axs[0].plot_surface(SY, N,  abs(rf2), cmap=cm.coolwarm)
surf = axs[1].plot_surface(SY, N,  abs(rf3), cmap=cm.coolwarm)
axs[0].set_xlim([0.1, 0.5])
axs[1].set_xlim([0.1, 0.5])
axs[0].set_ylim([0, 2000])
axs[1].set_ylim([0, 2000])
# ax.set_zlim([0, 1.5])
axs[0].set_ylabel("n [-]")
axs[1].set_ylabel("n [-]")
axs[0].set_xlabel(r"$\sigma_y [MPa]$")
axs[1].set_xlabel(r"$\sigma_y [MPa]$")
axs[0].set_zlabel(r"$F_n [N]$", rotation=90)
axs[1].set_zlabel(r"$F_t [N]$", rotation=90)
# fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)


# %%
fileNameID = "SY"+str(250)+"_n0"+str(int(10*0.1)) + \
    "_d0050_E200000_mu00_rho78_Poisson03"
path = "ScratchData/MaterialSweep/"
rfs, coords = LoadScratchTestData(
    fileNameID=fileNameID, path=path, toLoad="both")
x, y, z = coords[:, 3], coords[:, 4], coords[:, 5]
TopographyPlot(z, x, y, title="", modelSize="half")

print(np.unique(coords[:, 0]))

# %%
plt.figure()

for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
    fileNameID = "SY"+str(200)+"_n0"+str(int(10*i)) + \
        "_d0050_E200000_mu00_rho78_Poisson03"
    path = "ScratchData/MaterialSweep/"
    rfs, coords = LoadScratchTestData(
        fileNameID=fileNameID, path=path, toLoad="both")
    x_plot, y_plot = GetScratchProfile(coords=coords)
    h_r, _, _ = GetTopographyData(coords, 1.5, 2.0)
    plt.plot(x_plot/h_r, (y_plot)/h_r, label="n="+str(i))

plt.xlim([0, 6])
plt.xlabel(r"Distance to scratch center normalised by $h_r$")
plt.ylabel(r"Surface height normalised by $h_r$")
plt.legend()
plt.show()
