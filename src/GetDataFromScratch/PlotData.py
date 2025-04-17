# %% Importing
import numpy as np
from GetDataFromScratch.GetTopographyData import LoadScratchTestData, GetTopographyData, GetScratchProfile
from GetDataFromScratch.TopographyPlot import TopographyPlot
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import seaborn as sns
np.set_printoptions(legacy='1.25')
# %%Extract basic features

yield_strengths = np.arange(50, 2050, 50)
strain_hardening_indexs = np.arange(0.1, 0.6, 0.1)
youngs_modulus = 200000

scratch_depth = 50e-3
indenter_angle = 120
indenter_radius_at_scratch_depth = scratch_depth / \
    np.tan((90 - indenter_angle/2)*np.pi/180)

# color = ["b", "r", "g", "y", "k"]

h_rs = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
ws = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
h_ps = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
rf2 = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
rf3 = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
sy_s = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
ns = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))

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
        h_r, w, h_p = GetTopographyData(coords, 2.00, 2.44)
        h_rs[i, j] = h_r
        ws[i, j] = w
        h_ps[i, j] = h_p
        rf2[i, j] = rfs[-4, 2]
        rf3[i, j] = rfs[-4, 3]
        sy_s[i, j] = sy
        ns[i, j] = round(n, 1)

H_s = abs(rf2)/(1/4*(ws)**2)
all_raw_data = {"Yield Strength": sy_s.reshape(-1, ),
                "Strain Hardening": ns.reshape(-1, ),
                "Residual Depth": h_rs.reshape(-1, ),
                "Width": ws.reshape(-1, ),
                "Pile-up Height": h_ps.reshape(-1, ),
                "Normal Force": abs(rf2).reshape(-1, ),
                "Tangential Force": rf3.reshape(-1, ), }
all_raw_data_pandas = pd.DataFrame(all_raw_data)

all_raw_data_pandas.to_csv(
    "ScratchData/MaterialSweep/AllRawData.csv", index=False)

# %% Dimensional analysis data
# Normal force normalisation
# projected_area_normal = 1/2*(2*r * r)
projected_area_normal = 1/2*(ws * ws/2)
# projected_area_normal = (ws * ws)
fnNorm = abs(rf2)/(youngs_modulus*projected_area_normal)

# Tangential force normalisation
projected_area_tangential = scratch_depth * np.sqrt(scratch_depth*ws)
# projected_area_tangential = 1/2*scratch_depth*(2*r)
# projected_area_tangential = 1/2*scratch_depth*(ws)
# projected_area_tangential = 1/2*h_rs*(ws)
# projected_area_tangential = 1/2*h_rs*(2*r)
ftNorm = rf3/(youngs_modulus*projected_area_tangential)

# Scratch width normalisation
# wNorm = ws/(r)
# wNorm = ws/(h_rs)
wNorm = ws/(scratch_depth)

# Yield stress normalisation
syNorm = sy_s/youngs_modulus

# Pile-up height normalisation
hpNorm = h_ps/scratch_depth
# hpNorm = h_ps/h_rs

# Residual depth normalisation
hrNorm = h_rs/scratch_depth

norm_data = {"syNorm": syNorm.reshape(-1,),
             "n": ns.reshape(-1,),
             "hrNorm": hrNorm.reshape(-1,),
             "wNorm": wNorm.reshape(-1,),
             "hpNorm": hpNorm.reshape(-1,),
             "fnNorm": fnNorm.reshape(-1,),
             "ftNorm": ftNorm.reshape(-1,), }

norm_data_pandas = pd.DataFrame(norm_data)
# norm_data_pandas.to_csv(
#     "ScratchData/MaterialSweep/NormData.csv", index=False)
# %% Plotting w/2 vs (h_p+h_d)/(tan(90-theta))
plt.figure()
plt.plot(h_ps, ws/2)

# %% Plotting sy/E vs w/(F_n/E)^0.5
plt.figure(1)
plt.plot(yield_strengths/youngs_modulus, ws / np.sqrt(abs(rf2)/youngs_modulus))
plt.vlines([600/200000, 600/200000], 0, 60, "k", "--")
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4",
           "n=0.5", r"$\sigma_y=600$ [MPa]"])
plt.xlabel("Yield strength / Youngs modulus [-]")
plt.ylabel(r"$w/(F_n/E)^{0.5}$ [-]")
plt.xlim([0, 0.01])
plt.ylim([0, 60])
print(ws[np.where(yield_strengths == 600)] /
      (abs(rf2[np.where(yield_strengths == 600)])/youngs_modulus)**0.5)

# %% Plotting sy vs w/(F_n/sy)^0.5
plt.figure(2)
plt.plot(yield_strengths, ws / (abs(rf2)/yield_strengths.reshape(-1, 1))**0.5)
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel("Yield strength [MPa]")
plt.ylabel(r"$w/(F_n/\sigma_y)^{0.5}$ [-]")

# %% plotting sy vs F_n/(Eh_r^2)
plt.figure(3)
plt.plot(yield_strengths, abs(rf2) / (youngs_modulus*h_rs**2))
plt.vlines([600, 600], 0, 1, "k", "--")
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4",
           "n=0.5", r"$\sigma_y=600$ [MPa]"])
plt.xlabel("Yield strength [MPa]")
plt.ylabel(r"$F_n/(Eh_r^2)$ [-]")
print(abs(rf2[np.where(yield_strengths == 600)]) /
      (youngs_modulus*h_rs[np.where(yield_strengths == 600)]**2))


# %% Plotting sy vs F_n/(Esy)^0.5h_r^2
plt.figure(4)
plt.plot(yield_strengths, abs(rf2) /
         ((youngs_modulus*yield_strengths.reshape(-1, 1))**0.5*h_rs**2))
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel("Yield strength [MPa]")
plt.ylabel(r"$F_n/(E\sigma_y)^{0.5}h_r^2$ [-]")

# %% plotting n vs F_t/F_n
plt.figure(5)
plt.plot(strain_hardening_indexs, rf3[1:-1:5, :].T/abs(rf2[1:-1:5, :].T))
plt.legend(yield_strengths[1:-1:5])
plt.xlabel(r"n [-]")
plt.ylabel(r"$F_t/F_n$ [-]")
plt.ylim([0, 0.4])

# %% plotting sy/E vs h_p/h_r
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
E_star = ((1-0.3**2)/youngs_modulus + (1-0.2**2)/1220000)**(-1)
plt.plot(yield_strengths/(youngs_modulus), h_ps/h_rs, "o-")
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel(r"$\sigma_y/E$ [N]")
ax.set_xscale("log")
plt.ylabel(r"$h_p/h_r$ [-]")
plt.ylim([0, 1])

# %% plotting w vs h_p/h_r
fig = plt.figure()
plt.plot(ws, h_ps/h_rs, "o-")
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel(r"$w$ [mm]")
# ax.set_xscale("log")
plt.ylabel(r"$h_p$ [mm]")
# plt.ylim([0, 1])

# %% plotting sy/E vs H_s/sy
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(yield_strengths/youngs_modulus, H_s /
         yield_strengths.reshape(-1, 1), "o-")
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel(r"$\sigma_y/E$ [-]")
ax.set_xscale("log")
ax.set_yscale("log")
plt.ylabel(r"$H_s/\sigma_y$ [-]")
plt.ylim([2, 200])
plt.xlim([0.0001, 0.1])

# %% plotting N, SY vs F_n/(Esy)^0.5h_r^2
fnNorm = abs(rf2)/((youngs_modulus*yield_strengths.reshape(-1, 1))**0.5*h_rs**2)
SY, N = np.meshgrid(strain_hardening_indexs, yield_strengths)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
surf = ax.plot_surface(N, SY,  fnNorm, cmap=cm.coolwarm)
ax.set_ylabel("n [-]")
ax.set_xlabel(r"$\sigma_y [MPa]$")
ax.set_zlabel(r"$F_n/((E\sigma_y)^{0.5}h_r^2) [-]$", rotation=90)
ax.set_xlim([2000, 0])
ax.set_ylim([0.1, 0.5])
ax.set_zlim([0, 12])

fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
plt.show()
# %% plottig SY, N vs w/(F_n/sy)^0.5
wNorm = ws / (abs(rf2)/yield_strengths.reshape(-1, 1))**0.5
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
surf = ax.plot_surface(SY, N,  wNorm, cmap=cm.coolwarm)
ax.set_xlabel("n [-]")
ax.set_ylabel(r"$\sigma_y [MPa]$")
ax.set_zlabel(r"$w/(F_n/\sigma_y)^{0.5} [-]$", rotation=90)

ax.set_xlim([0.5, 0.1])
ax.set_ylim([0, 2000])
ax.set_zlim([0, 1.5])
fig.colorbar(surf, shrink=0.5, aspect=5)

# %% plotting SY, N vs F_n
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(10, 10))
SY, N = np.meshgrid(strain_hardening_indexs, yield_strengths/1000)
plt.subplots_adjust(left=None, bottom=None, right=None,
                    top=None, wspace=None, hspace=None)
surf = ax.plot_surface(SY, N,  abs(rf2), cmap=cm.coolwarm)
ax.set_xlim([0.1, 0.5])
ax.set_ylim([0, 2])
ax.set_xlabel("n [-]")
ax.set_ylabel(r"$\sigma_y [GPa]$")
ax.set_zlabel(r"$F_n [N]$", rotation=90)
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
# %% plotting SY, N vs  F_t
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(10, 10))
SY, N = np.meshgrid(strain_hardening_indexs, yield_strengths/1000)
plt.subplots_adjust(left=None, bottom=None, right=None,
                    top=None, wspace=None, hspace=None)
surf = ax.plot_surface(SY, N, rf3, cmap=cm.coolwarm)
ax.set_xlim([0.1, 0.5])
ax.set_ylim([0, 2])
ax.set_xlabel("n [-]")
ax.set_ylabel(r"$\sigma_y [GPa]$")
ax.set_zlabel(r"$F_t [N]$", rotation=90)
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)


# %% Plotting topography
fileNameID = "SY"+str(2000)+"_n0"+str(int(10*0.1)) + \
    "_d0050_E200000_mu00_rho78_Poisson03"
path = "ScratchData/Data1/"
rfs, coords = LoadScratchTestData(
    fileNameID=fileNameID, path=path, toLoad="both")
x, y, z = coords[:, 3], coords[:, 4], coords[:, 5]
TopographyPlot(z, x, y, title="", modelSize="half")
h_r, w, h_p = GetTopographyData(coords, 2.00, 2.44)
print(h_r, w, h_p)
# plt.figure()
# plt.plot(z, y)
# print(np.unique(coords[:, 0]))

# %% Plotting scratch profile
plt.figure()

for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
    fileNameID = "SY"+str(2000)+"_n0"+str(int(10*i)) + \
        "_d0050_E200000_mu00_rho78_Poisson03"
    path = "ScratchData/MaterialSweep/"
    rfs, coords = LoadScratchTestData(
        fileNameID=fileNameID, path=path, toLoad="both")
    x_plot, y_plot = GetScratchProfile(
        coords=coords, lowerBound=2.00, upperBound=2.44)
    h_r, w, h_p = GetTopographyData(coords, 2.0, 2.44)
    # print(h_r)
    plt.plot(x_plot/h_r, (y_plot)/h_r, "o-", label="n="+str(i))

plt.xlim([0, 6])
plt.ylim([-1, 0.8])
plt.xlabel(r"Distance to scratch center normalised by $h_r$")
plt.ylabel(r"Surface height normalised by $h_r$")
plt.legend()
plt.show()

# %% Plotting scratch profile and comparison for different meshes
plt.figure()

meshes = ["X002_Y0025_Z004", "X002_Y0025_Z002", "X002_Y001_Z002",
          "X001_Y001_Z002", "X001_Y001_Z001", "X001_Y0005_Z001",
          "X0005_Y001_Z001", "X001_Y001_Z0005"]
w_s = []
h_rs = []
h_ps = []
h_rps = []
maxRF2 = []
maxRF3 = []
for mesh in meshes:
    fileNameID = mesh+"_MassScaling104_Velocity2mPERs"
    path = "ScratchData/MeshDependenceData/"
    rfs, coords = LoadScratchTestData(
        fileNameID=fileNameID, path=path, toLoad="both")
    x_plot, y_plot = GetScratchProfile(
        coords=coords, lowerBound=2.00, upperBound=2.44)
    h_r, w, h_p = GetTopographyData(coords, 2.00, 2.44)
    plt.plot(x_plot/h_r, (y_plot)/h_r, label=mesh)
    h_rp = h_r+h_p
    w_s.append(round(w, 5))
    h_rs.append(round(h_r, 5))
    h_ps.append(round(h_p, 5))
    h_rps.append(round(h_rp, 5))
    maxRF2.append(max(abs(rfs[:-2, 2])))
    maxRF3.append(max(rfs[:-2, 3]))
plt.xlim([0, 6])
plt.xlabel(r"Distance to scratch center normalised by $h_r$")
plt.ylabel(r"Surface height normalised by $h_r$")
plt.legend()
# plt.grid()
plt.show()
print(meshes)
print(w_s)
print(h_rs)
print(h_ps)
print(h_rps)
print(maxRF2)
print(maxRF3)
# %% Plotting scratch profile and comparison for different mass scalings
plt.figure()

massScale = [100, 101, 102, 103, 104, 105]

w_s = []
h_rs = []
h_ps = []
h_rps = []
maxRF2 = []
maxRF3 = []
for mass in massScale:
    fileNameID = "X002_Y0025_Z004_MassScaling"+str(mass)+"_Velocity02mPERs"
    path = "ScratchData/MassScaleTestingData/"
    rfs, coords = LoadScratchTestData(
        fileNameID=fileNameID, path=path, toLoad="both")
    x_plot, y_plot = GetScratchProfile(
        coords=coords, lowerBound=2.00, upperBound=2.44)
    h_r, w, h_p = GetTopographyData(coords, 2.00, 2.44)
    plt.plot(x_plot/h_r, (y_plot)/h_r, label=str(mass))
    h_rp = h_r+h_p
    w_s.append(round(w, 5))
    h_rs.append(round(h_r, 5))
    h_ps.append(round(h_p, 5))
    h_rps.append(round(h_rp, 5))
    maxRF2.append(max(abs(rfs[:-2, 2])))
    maxRF3.append(max(rfs[:-2, 3]))
plt.xlim([0, 6])
plt.xlabel(r"Distance to scratch center normalised by $h_r$")
plt.ylabel(r"Surface height normalised by $h_r$")
plt.legend()
# plt.grid()
plt.show()
print(massScale)
print(w_s)
print(h_rs)
print(h_ps)
print(h_rps)
print(maxRF2)
print(maxRF3)

# %% Plotting scratch profile and comparison for different velocities
plt.figure()

velocities = ["002", "02", "2"]

w_s = []
h_rs = []
h_ps = []
h_rps = []
maxRF2 = []
maxRF3 = []
for velocity in velocities:
    fileNameID = "X002_Y0025_Z004_MassScaling104_Velocity"+velocity+"mPERs"
    path = "ScratchData/VelocityTestingData/"
    rfs, coords = LoadScratchTestData(
        fileNameID=fileNameID, path=path, toLoad="both")
    x_plot, y_plot = GetScratchProfile(
        coords=coords, lowerBound=2.00, upperBound=2.44)
    h_r, w, h_p = GetTopographyData(coords, 2.00, 2.44)
    plt.plot(x_plot/h_r, (y_plot)/h_r, label=velocity)
    h_rp = h_r+h_p
    w_s.append(round(w, 5))
    h_rs.append(round(h_r, 5))
    h_ps.append(round(h_p, 5))
    h_rps.append(round(h_rp, 5))
    maxRF2.append(max(abs(rfs[:-2, 2])))
    maxRF3.append(max(rfs[:-2, 3]))
plt.xlim([0, 6])
plt.xlabel(r"Distance to scratch center normalised by $h_r$")
plt.ylabel(r"Surface height normalised by $h_r$")
plt.legend()
# plt.grid()
plt.show()
print(velocities)
print(w_s)
print(h_rs)
print(h_ps)
print(h_rps)
print(maxRF2)
print(maxRF3)
# %% plot sns.pairplot
sns.set_theme()
# sns.pairplot(all_raw_data_pandas, hue="Strain Hardening")
sns.pairplot(Fn_norm_data_pandas, hue="n")

# %%
sns.relplot(all_raw_data, x="sy", y="h_r", hue="n")
# sns.scatterplot(all_norm_data, x="S_y/E", y="h_p/h_r", hue="n")
