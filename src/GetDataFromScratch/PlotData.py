# %% Importing
import numpy as np
from GetDataFromScratch.GetTopographyData import (
    LoadScratchTestData,
    GetTopographyData,
    GetScratchProfile,
)
from GetDataFromScratch.TopographyPlot import TopographyPlot
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata

from matplotlib import rcParams

rcParams.update(
    {
        "font.family": "serif",  # or "DejaVu Serif"
        "mathtext.fontset": "cm",  # Use Computer Modern (LaTeX default)
        "mathtext.rm": "serif",
        "font.size": 12,
    }
)


np.set_printoptions(legacy="1.25")
# %%Extract basic features

yield_strengths = np.arange(50, 2050, 50)
strain_hardening_indexs = np.arange(0.1, 0.6, 0.1)
youngs_modulus = 200000

scratch_depth = 50e-3
indenter_angle = 120
indenter_radius_at_scratch_depth = scratch_depth / np.tan(
    (90 - indenter_angle / 2) * np.pi / 180
)

# color = ["b", "r", "g", "y", "k"]

h_rs = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
ws = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
h_ps = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
rf2 = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
rf3 = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
rf2_test = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
rf3_test = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
sy_s = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))
ns = np.zeros((len(yield_strengths), len(strain_hardening_indexs)))

for i, sy in enumerate(yield_strengths):
    count = 0
    for j, n in enumerate(strain_hardening_indexs):
        # print(yield_strengths)
        fileNameID = (
            "SY"
            + str(sy)
            + "_n0"
            + str(int(10 * n))
            + "_d0050_E200000_mu00_rho78_Poisson03"
        )
        path = "ScratchData/MaterialSweep/"
        rfs, coords = LoadScratchTestData(
            fileNameID=fileNameID, path=path, toLoad="both"
        )
        # x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        h_r, w, h_p = GetTopographyData(coords, 2.00, 2.44)
        h_rs[i, j] = h_r
        ws[i, j] = w
        h_ps[i, j] = h_p
        rf2_test[i, j] = np.mean(rfs[-2 - 2 : -2, 2])
        rf3_test[i, j] = np.mean(rfs[-2 - 2 : -2, 3])
        rf2[i, j] = abs(rfs[-4, 2])
        rf3[i, j] = rfs[-4, 3]
        sy_s[i, j] = sy
        ns[i, j] = round(n, 1)

H_s = abs(rf2) / (1 / 4 * (ws) ** 2)
all_raw_data = {
    "Yield Strength": sy_s.reshape(
        -1,
    ),
    "Strain Hardening": ns.reshape(
        -1,
    ),
    "Residual Depth": h_rs.reshape(
        -1,
    ),
    "Width": ws.reshape(
        -1,
    ),
    "Pile-up Height": h_ps.reshape(
        -1,
    ),
    "Normal Force": abs(rf2).reshape(
        -1,
    ),
    "Tangential Force": rf3.reshape(
        -1,
    ),
}
all_raw_data_pandas = pd.DataFrame(all_raw_data)

# all_raw_data_pandas.to_csv("ScratchData/MaterialSweep/AllRawData.csv", index=False)


# %% Dimensional analysis data
# Normal force normalisation
# projected_area_normal = 1/2*(2*r * r)
projected_area_normal = 1 / 2 * (ws * ws / 2)
# projected_area_normal = (ws * ws)
fnNorm = abs(rf2) / (youngs_modulus * projected_area_normal)

# Tangential force normalisation
projected_area_tangential = scratch_depth * np.sqrt(scratch_depth * ws)
# projected_area_tangential = 1/2*scratch_depth*(2*r)
# projected_area_tangential = 1/2*scratch_depth*(ws)
# projected_area_tangential = 1/2*h_rs*(ws)
# projected_area_tangential = 1/2*h_rs*(2*r)
ftNorm = rf3 / (youngs_modulus * projected_area_tangential)

# Scratch width normalisation
# wNorm = ws/(r)
# wNorm = ws/(h_rs)
wNorm = ws / (scratch_depth)

# Yield stress normalisation
syNorm = sy_s / youngs_modulus

# Pile-up height normalisation
hpNorm = h_ps / scratch_depth
# hpNorm = h_ps/h_rs

# Residual depth normalisation
hrNorm = h_rs / scratch_depth

norm_data = {
    "syNorm": syNorm.reshape(
        -1,
    ),
    "n": ns.reshape(
        -1,
    ),
    "hrNorm": hrNorm.reshape(
        -1,
    ),
    "wNorm": wNorm.reshape(
        -1,
    ),
    "hpNorm": hpNorm.reshape(
        -1,
    ),
    "fnNorm": fnNorm.reshape(
        -1,
    ),
    "ftNorm": ftNorm.reshape(
        -1,
    ),
}

norm_data_pandas = pd.DataFrame(norm_data)
# norm_data_pandas.to_csv(
#     "ScratchData/MaterialSweep/NormData.csv", index=False)


# %% Zhang validation data
zhang_data = {
    "syNorm": syNorm.reshape(
        -1,
    ),
    "n": ns.reshape(
        -1,
    ),
    "Fn/(sqrt(E*sy)h_r^2)": (
        abs(rf2) / ((youngs_modulus * sy_s) ** 0.5 * h_rs**2)
    ).reshape(
        -1,
    ),
    "w/(sqrt(Fn/sy))": (ws / (np.sqrt(abs(rf2) / sy_s))).reshape(
        -1,
    ),
}
zhang_data_pandas = pd.DataFrame(zhang_data)
# zhang_data_pandas.to_csv("ScratchData/MaterialSweep/ZhangData.csv", index=False)

# %% New improved dimensionless groups

FtFn = abs(rf3) / abs(rf2)
hphr = h_ps / h_rs
HsE = H_s / youngs_modulus

norm_data = {
    "syNorm": syNorm.reshape(
        -1,
    ),
    "n": ns.reshape(
        -1,
    ),
    "Ft/Fn": FtFn.reshape(
        -1,
    ),
    "h_p/h_r": hphr.reshape(
        -1,
    ),
    "H_s/E": HsE.reshape(
        -1,
    ),
}


norm_data_pandas = pd.DataFrame(norm_data)
# norm_data_pandas.to_csv("ScratchData/MaterialSweep/NewNormData.csv", index=False)

# %% plot F_t/F_n vs sy
plt.figure()

# plt.plot(yield_strengths, abs(rf2), c="b")
# plt.plot(yield_strengths, rf3, c="k")

plt.plot(yield_strengths / youngs_modulus, rf3 / abs(rf2), "o", markersize=3)
plt.plot(yield_strengths / youngs_modulus)
plt.legend([r"$n=0.1$", r"$n=0.2$", r"$n=0.3$", r"$n=0.4$", r"$n=0.5$"])
plt.xlabel(r"$\sigma_y/E$ [-]")
plt.ylabel(r"$F_t/F_n$ [-]")
plt.grid()
plt.ylim([0, 0.4])
# plt.xlim([0, 0.01])


# %% Plotting H_s/E vs sy in log-log scale.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(sy_s, norm_data["H_s/E"].reshape(40, 5))
ax.set_yscale("log")
ax.set_xscale("log")

# %% Plotting sy/E vs w/(F_n/E)^0.5
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(yield_strengths / youngs_modulus, norm_data["h_p/h_r"].reshape(40, 5), "x-")
# plt.vlines([600 / 200000, 600 / 200000], 0, 60, "k", "--")
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel("Yield strength / Youngs modulus [-]")
plt.ylabel(r"$h_p/h_r$ [-]")
# ax.set_yscale("log")
ax.set_xscale("log")
# plt.xlim([0, 0.01])
# plt.ylim([0, 60])


# %% Plotting sy vs w/(F_n/sy)^0.5
plt.figure(2)
plt.plot(yield_strengths, ws / (abs(rf2) / yield_strengths.reshape(-1, 1)) ** 0.5)
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel("Yield strength [MPa]")
plt.ylabel(r"$w/(F_n/\sigma_y)^{0.5}$ [-]")

# %% plotting sy vs F_n/(Eh_r^2)
plt.figure(3)
plt.plot(yield_strengths, abs(rf2) / (youngs_modulus * h_rs**2))
plt.vlines([600, 600], 0, 1, "k", "--")
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5", r"$\sigma_y=600$ [MPa]"])
plt.xlabel("Yield strength [MPa]")
plt.ylabel(r"$F_n/(Eh_r^2)$ [-]")
print(
    abs(rf2[np.where(yield_strengths == 600)])
    / (youngs_modulus * h_rs[np.where(yield_strengths == 600)] ** 2)
)


# %% Plotting sy vs F_n/(Esy)^0.5h_r^2
plt.figure(4)
plt.plot(
    yield_strengths,
    abs(rf2) / ((youngs_modulus * yield_strengths.reshape(-1, 1)) ** 0.5 * h_rs**2),
)
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel("Yield strength [MPa]")
plt.ylabel(r"$F_n/(E\sigma_y)^{0.5}h_r^2$ [-]")

# %% plotting n vs F_t/F_n
plt.figure(5)
plt.plot(strain_hardening_indexs, rf3[1:-1:5, :].T / abs(rf2[1:-1:5, :].T))
plt.legend(yield_strengths[1:-1:5])
plt.xlabel(r"n [-]")
plt.ylabel(r"$F_t/F_n$ [-]")
plt.ylim([0, 0.4])

# %% plotting sy/E vs h_p/h_r
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
E_star = ((1 - 0.3**2) / youngs_modulus + (1 - 0.2**2) / 1220000) ** (-1)
plt.plot(yield_strengths / (youngs_modulus), h_ps / h_rs, "o-")
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel(r"$\sigma_y/E$ [N]")
ax.set_xscale("log")
plt.ylabel(r"$h_p/h_r$ [-]")
plt.ylim([0, 1])

# %% plotting w vs h_p/h_r
fig = plt.figure()
plt.plot(ws, h_ps / h_rs, "o-")
plt.legend(["n=0.1", "n=0.2", "n=0.3", "n=0.4", "n=0.5"])
plt.xlabel(r"$w$ [mm]")
# ax.set_xscale("log")
plt.ylabel(r"$h_p$ [mm]")
# plt.ylim([0, 1])

# %% plotting sy/E vs H_s/sy and comparing to Bellemare.

# Bellemare data
SyEn05 = np.array(
    [
        0.00009127550492545203,
        0.00018183662305265322,
        0.0003657836973637655,
        0.000910996230467547,
        0.0013829268154842353,
        0.001850441853902186,
        0.002755029865482474,
        0.0036507885630107538,
        0.004564064807152558,
    ]
)
SyEn035 = np.array(
    [
        0.00018361038117557183,
        0.0003657836973637655,
        0.0005552735201678112,
        0.000910996230467547,
        0.0013695671260338807,
        0.001832565760794961,
        0.002755029865482474,
        0.003686400838249849,
        0.004608585800169843,
        0.005488496914453607,
        0.00734394929059335,
        0.009181101541270305,
    ]
)
SyEn02 = np.array(
    [
        0.0003657836973637655,
        0.0005499093304605992,
        0.0009198827074412861,
        0.0013829268154842353,
        0.001832565760794961,
        0.002755029865482474,
        0.0036507885630107538,
        0.004564064807152558,
        0.005488496914453607,
        0.00734394929059335,
        0.009181101541270305,
        0.01193227932178964,
        0.014630419628596862,
        0.018290345274250854,
    ]
)
SyEn01 = np.array(
    [
        0.000910996230467547,
        0.0013829268154842353,
        0.001850441853902186,
        0.002755029865482474,
        0.0036507885630107538,
        0.004564064807152558,
        0.005488496914453607,
        0.00734394929059335,
        0.009181101541270305,
        0.012048674891706431,
        0.014773134694584501,
        0.01846876174479757,
        0.024003048313526922,
        0.029430644365749936,
    ]
)
SyEn002 = np.array(
    [
        0.004564064807152558,
        0.005596096231070952,
        0.007273003467023406,
        0.009270660251516773,
        0.011817008184959324,
        0.014630419628596862,
        0.01846876174479757,
        0.024003048313526922,
        0.029430644365749936,
        0.03715187756641498,
        0.0646079010962975,
    ]
)
H_sEn05 = np.array(
    [
        127.24072146061593,
        91.9490580850864,
        64.36313475662794,
        41.20971081286682,
        32.14517167233989,
        27.413182429943543,
        22.50119858984532,
        19.936406019748024,
        17.55180701918815,
    ]
)
H_sEn035 = np.array(
    [
        31.941072734389245,
        25.55822488976337,
        22.216372531184856,
        19.188868179745473,
        16.260234871130816,
        14.873024394737772,
        12.846224294454649,
        11.527874003530679,
        10.544395737996581,
        9.956929482342783,
        8.878363054142227,
        8.43728824762442,
    ]
)
H_sEn02 = np.array(
    [
        9.224236102937128,
        8.7103206942872,
        7.8164196103467924,
        7.3809387278754945,
        6.794387742987376,
        6.498102619087817,
        6.019931434271516,
        5.8312315383745315,
        5.684539377025377,
        5.333751302423667,
        5.367833218994537,
        5.004610060594387,
        4.941260397108687,
        4.60690198195443,
    ]
)
H_sEn01 = np.array(
    [
        4.213873053176194,
        4.081785600829043,
        3.8543746269920325,
        3.903789819866838,
        3.7335559600682307,
        3.709850539733365,
        3.593562103845928,
        3.525546113450701,
        3.593562103845928,
        3.5031614074430344,
        3.4588174738277044,
        3.415034857325627,
        3.525546113450701,
        3.5031614074430344,
    ]
)
H_sEn002 = np.array(
    [
        2.483605165552323,
        2.4994750513508737,
        2.467836042221924,
        2.6638595513143373,
        2.6638595513143373,
        2.630139720177506,
        2.6469459410585,
        2.7676351255903446,
        2.750062614373864,
        2.80311772309926,
        2.750062614373864,
    ]
)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
c = np.array(["k", "r", "g", "b", "m"])
c_bellemare = np.array(["k", "y", "b", "m", "c"])
plt.plot(
    yield_strengths / youngs_modulus,
    H_s[:, 4] / yield_strengths,
    "o",
    c=c[0],
    label=r"FEM $n=0.5$",
)
plt.plot(SyEn05, H_sEn05, "x", c=c_bellemare[0], label="Bellemare $n=0.5$")

plt.plot(
    yield_strengths / youngs_modulus,
    H_s[:, 3] / yield_strengths,
    "o",
    c=c[1],
    label=r"FEM $n=0.4$",
)
plt.plot(SyEn035, H_sEn035, "x", c=c_bellemare[1], label="Bellemare $n=0.35$")

plt.plot(
    yield_strengths / youngs_modulus,
    H_s[:, 2] / yield_strengths,
    "o",
    c=c[2],
    label=r"FEM $n=0.3$",
)
plt.plot(SyEn02, H_sEn02, "x", c=c_bellemare[2], label="Bellemare $n=0.2$")

plt.plot(
    yield_strengths / youngs_modulus,
    H_s[:, 1] / yield_strengths,
    "o",
    c=c[3],
    label=r"FEM $n=0.2$",
)
plt.plot(SyEn01, H_sEn01, "x", c=c_bellemare[3], label="Bellemare $n=0.1$")

plt.plot(
    yield_strengths / youngs_modulus,
    H_s[:, 0] / yield_strengths,
    "o",
    c=c[4],
    label=r"FEM $n=0.1$",
)
plt.plot(SyEn002, H_sEn002, "x", c=c_bellemare[4], label="Bellemare $n=0.02$")
plt.legend(
    # loc="upper center",
    # bbox_to_anchor=(0.5, 1.15),
    ncol=1,
    fancybox=True,
    shadow=True,
    fontsize="small",
)
plt.xlabel(r"$\sigma_y/E$ [-]")
ax.set_xscale("log")
ax.set_yscale("log")
plt.ylabel(r"$H_s/\sigma_y$ [-]")
plt.ylim([2, 200])
plt.xlim([0.00005, 0.1])
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontfamily("STIXGeneral")
plt.grid()

# %% plotting N, SY vs F_n/(Esy)^0.5h_r^2
fnNorm = abs(rf2) / ((youngs_modulus * yield_strengths.reshape(-1, 1)) ** 0.5 * h_rs**2)
SY, N = np.meshgrid(strain_hardening_indexs, yield_strengths)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
surf = ax.plot_surface(N, SY, fnNorm, cmap=cm.viridis, antialiased=True)
ax.set_ylabel("n [-]")
ax.set_xlabel(r"$\sigma_y [MPa]$")
ax.set_zlabel(r"$F_n/((E\sigma_y)^{0.5}h_r^2) [-]$", rotation=90)
ax.set_xlim([2000, 0])
ax.set_ylim([0.1, 0.5])
ax.set_zlim([0, 12])


fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
plt.show()
# %% plottig SY, N vs w/(F_n/sy)^0.5
wNorm = ws / (abs(rf2) / yield_strengths.reshape(-1, 1)) ** 0.5
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
surf = ax.plot_surface(SY, N, wNorm, cmap=cm.coolwarm)
ax.set_xlabel("n [-]")
ax.set_ylabel(r"$\sigma_y [MPa]$")
ax.set_zlabel(r"$w/(F_n/\sigma_y)^{0.5} [-]$", rotation=90)

ax.set_xlim([0.5, 0.1])
ax.set_ylim([0, 2000])
ax.set_zlim([0, 1.5])
fig.colorbar(surf, shrink=0.5, aspect=5)

# %% plotting SY, N vs F_n
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(10, 10))
SY, N = np.meshgrid(strain_hardening_indexs, yield_strengths)
plt.subplots_adjust(
    left=None, bottom=None, right=None, top=None, wspace=None, hspace=None
)
surf = ax.plot_surface(N, SY, h_ps / h_rs, cmap=cm.viridis)
ax.set_ylim([0.1, 0.5])
# ax.set_xlim([0, 2000])
ax.set_ylabel("n [-]")
ax.set_xlabel(r"$\sigma_y [MPa]$")
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel(r"$w [mm]$", rotation=90)
ax.view_init(azim=45, elev=30)
# ax.xaxis._axinfo["label"]["space_factor"] = 5
ax.set_box_aspect(None, zoom=0.90)


fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.01)


# %% plotting SY, N vs  F_t
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(10, 10))
SY, N = np.meshgrid(strain_hardening_indexs, yield_strengths / 1000)
plt.subplots_adjust(
    left=None, bottom=None, right=None, top=None, wspace=None, hspace=None
)
surf = ax.plot_surface(SY, N, rf3, cmap=cm.coolwarm)
ax.set_xlim([0.1, 0.5])
ax.set_ylim([0, 2])
ax.set_xlabel("n [-]")
ax.set_ylabel(r"$\sigma_y [GPa]$")
ax.set_zlabel(r"$F_t [N]$", rotation=90)
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)


# %% Plotting topography
fileNameID = (
    "SY" + str(150) + "_n0" + str(int(10 * 0.1)) + "_d0050_E200000_mu00_rho78_Poisson03"
)
path = "ScratchData/MaterialSweep/"
rfs, coords = LoadScratchTestData(fileNameID=fileNameID, path=path, toLoad="both")
x, y, z = coords[:, 3], coords[:, 4], coords[:, 5]
TopographyPlot(z, x, y, title="", modelSize="half")
h_r, w, h_p = GetTopographyData(coords, 2.00, 2.44)
print(h_r, w, h_p)
# plt.figure()
# plt.plot(z, y)
# print(np.unique(coords[:, 0]))

# %% Plotting scratch profile for one SY all n
plt.figure()

for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
    fileNameID = (
        "SY"
        + str(100)
        + "_n0"
        + str(int(10 * i))
        + "_d0050_E200000_mu00_rho78_Poisson03"
    )
    path = "ScratchData/MaterialSweep/"
    rfs, coords = LoadScratchTestData(fileNameID=fileNameID, path=path, toLoad="both")
    x_plot, y_plot = GetScratchProfile(coords=coords, lowerBound=2.00, upperBound=2.44)
    h_r, w, h_p = GetTopographyData(coords, 2.0, 2.44)
    # print(h_r)
    plt.plot(x_plot / h_r, (y_plot) / h_r, "o-", label="n=" + str(i))

plt.xlim([0, 7])
plt.ylim([-1, 0.8])
plt.xlabel(r"Distance to scratch center normalised by $h_r$")
plt.ylabel(r"Surface height normalised by $h_r$")
plt.legend()
plt.grid()
plt.show()
# %% Plotting scratch profile for one n all SY
plt.figure()
ax = plt.subplot(111)

for i in yield_strengths:
    fileNameID = "SY" + str(i) + "_n01_d0050_E200000_mu00_rho78_Poisson03"
    path = "ScratchData/MaterialSweep/"
    rfs, coords = LoadScratchTestData(fileNameID=fileNameID, path=path, toLoad="both")
    x_plot, y_plot = GetScratchProfile(coords=coords, lowerBound=2.00, upperBound=2.44)
    h_r, w, h_p = GetTopographyData(coords, 2.0, 2.44)
    # print(h_r)
    ax.plot(x_plot / h_r, (y_plot) / h_r, "o-", label="sy=" + str(i))
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5
)
plt.xlim([0, 7])
plt.ylim([-1, 0.8])
plt.xlabel(r"Distance to scratch center normalised by $h_r$")
plt.ylabel(r"Surface height normalised by $h_r$")
# plt.legend(lo)
plt.grid()
plt.show()

# %% Plotting scratch profile and comparison for different meshes
plt.figure()

meshes = [
    "X002_Y0025_Z004",
    "X002_Y0025_Z002",
    "X002_Y001_Z002",
    "X001_Y001_Z002",
    "X001_Y001_Z001",
    "X001_Y0005_Z001",
    "X0005_Y001_Z001",
    "X001_Y001_Z0005",
]
w_s = []
h_rs = []
h_ps = []
h_rps = []
maxRF2 = []
maxRF3 = []
for mesh in meshes:
    fileNameID = mesh + "_MassScaling104_Velocity2mPERs"
    path = "ScratchData/MeshDependenceData/"
    rfs, coords = LoadScratchTestData(fileNameID=fileNameID, path=path, toLoad="both")
    x_plot, y_plot = GetScratchProfile(coords=coords, lowerBound=2.00, upperBound=2.44)
    h_r, w, h_p = GetTopographyData(coords, 2.00, 2.44)
    plt.plot(x_plot / h_r, (y_plot) / h_r, label=mesh)
    h_rp = h_r + h_p
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
    fileNameID = "X002_Y0025_Z004_MassScaling" + str(mass) + "_Velocity02mPERs"
    path = "ScratchData/MassScaleTestingData/"
    rfs, coords = LoadScratchTestData(fileNameID=fileNameID, path=path, toLoad="both")
    x_plot, y_plot = GetScratchProfile(coords=coords, lowerBound=2.00, upperBound=2.44)
    h_r, w, h_p = GetTopographyData(coords, 2.00, 2.44)
    plt.plot(x_plot / h_r, (y_plot) / h_r, label=str(mass))
    h_rp = h_r + h_p
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
    fileNameID = "X002_Y0025_Z004_MassScaling104_Velocity" + velocity + "mPERs"
    path = "ScratchData/VelocityTestingData/"
    rfs, coords = LoadScratchTestData(fileNameID=fileNameID, path=path, toLoad="both")
    x_plot, y_plot = GetScratchProfile(coords=coords, lowerBound=2.00, upperBound=2.44)
    h_r, w, h_p = GetTopographyData(coords, 2.00, 2.44)
    plt.plot(x_plot / h_r, (y_plot) / h_r, label=velocity)
    h_rp = h_r + h_p
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
sns.pairplot(norm_data_pandas, hue="n")

# %% material behavior plots

yield_strength = [100]
n = [0.1]
plt.figure()
for i in yield_strength:
    for j in n:
        total_strain_data = np.append(
            np.arange(0.0001, 0.1, 0.001), np.linspace(0.1, 7, 200)
        )
        plastic_behaviour = []
        yield_strain = i / youngs_modulus
        K = youngs_modulus * yield_strain ** (1 - j)
        plastic_behaviour.append([i, 0])
        for total_strain in total_strain_data:
            if total_strain > yield_strain:
                yield_strength_ = round(K * total_strain**j, 5)
                plastic_strain = round(
                    total_strain - yield_strength_ / youngs_modulus, 5
                )
                plastic_behaviour.append([yield_strength_, plastic_strain])

        plastic_behaviour = np.array(plastic_behaviour)

        plt.plot(
            plastic_behaviour[:, 1], plastic_behaviour[:, 0], label=f"sy:{i}, n:{j}"
        )

# plt.xlim([0, 0.1])
# plt.ylim([0,3000])
plt.legend()
plt.show()
# %%
fileNameID1 = "ExplicitRigidIndenterScratch_test1"
fileNameID2 = "ExplicitRigidIndenterScratch_test2"
fileNameID3 = "ExplicitRigidIndenterScratch_test3"
path = "ScratchData/TestData/"
rfs1, coords1 = LoadScratchTestData(fileNameID=fileNameID1, path=path, toLoad="both")
h_r1, w1, h_p1 = GetTopographyData(coords1, 2.00, 2.44)
rfs2, coords2 = LoadScratchTestData(fileNameID=fileNameID2, path=path, toLoad="both")
h_r2, w2, h_p2 = GetTopographyData(coords2, 2.00, 2.44)
rfs3, coords3 = LoadScratchTestData(fileNameID=fileNameID3, path=path, toLoad="both")
h_r3, w3, h_p3 = GetTopographyData(coords3, 2.00, 2.44)

print(h_r1, h_r2, h_r3)
print(h_p1, h_p2, h_p3)

# %% showcase raw coordinate and force data


path = "ScratchData/MaterialSweep/"
# path = "ScratchData/TestData/"
fileNameID1 = "SY600_n02_d0050_E200000_mu00_rho78_Poisson03"
# fileNameID1 = "ExplicitRigidIndenterScratch"

rfs1, coords1 = LoadScratchTestData(fileNameID=fileNameID1, path=path, toLoad="both")
# h_r1, w1, h_p1 = GetTopographyData(coords1, 2.00, 2.44)
x, y, z = coords1[:, 3], coords1[:, 4], coords1[:, 5]

h_r, w, h_p = GetTopographyData(coords1, 2.00, 2.44)
print(h_r, w, h_p)
# # Extend data to full domain
z = np.append(z, z)
y = np.append(y, y)
x = np.append(x, -x)

y = y - 0.64
# Create a grid to interpolate onto
xi = np.linspace(z.min(), z.max(), 1000)
yi = np.linspace(x.min(), x.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate z values onto grid
zi = griddata((z, x), y, (xi, yi), method="cubic")

plt.figure(figsize=(8, 6))
# plt.axis("equal")
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
contour = plt.contourf(xi, yi, zi, levels=100, cmap="coolwarm")
# contour = plt.contourf(xi[:,:-200], yi[:,:-200], zi[:,:-200], levels=100, cmap="coolwarm")
cbar = plt.colorbar(
    contour,
    aspect=20,
    location="top",
    label="y-direction [mm]",
    # ticks=[round(y.min(),3),0, round(y.max(),3)]
)


# plt.xlim([0, 2.88])
# plt.ylim([-0.32, 0.32])
# plt.xticks([0, 1.00, 2.00, 2.88])
# plt.yticks([-0.32, 0, 0.32])
plt.xlim([2.00, 2.44])
plt.ylim([0, 0.32])
plt.xticks([2.00, 2.11, 2.22, 2.33, 2.44])
plt.yticks([0, 0.10, 0.21, 0.32])
for label in ax.get_xticklabels() + ax.get_yticklabels() + cbar.ax.get_xticklabels():
    label.set_fontfamily("STIXGeneral")

plt.xlabel("z-direction [mm]")
plt.ylabel("x-direction [mm]")
plt.show()

# %%
yield_strengths_test = np.arange(100, 2050, 100)
strain_hardening_indexs_test = np.arange(0.1, 0.6, 0.2)
youngs_modulus = 200000

h_rs = np.zeros((len(yield_strengths_test), len(strain_hardening_indexs_test)))
ws = np.zeros((len(yield_strengths_test), len(strain_hardening_indexs_test)))
h_ps = np.zeros((len(yield_strengths_test), len(strain_hardening_indexs_test)))
rf2 = np.zeros((len(yield_strengths_test), len(strain_hardening_indexs_test)))
rf3 = np.zeros((len(yield_strengths_test), len(strain_hardening_indexs_test)))
rf2_test = np.zeros((len(yield_strengths_test), len(strain_hardening_indexs_test)))
rf3_test = np.zeros((len(yield_strengths_test), len(strain_hardening_indexs_test)))
sy_s = np.zeros((len(yield_strengths_test), len(strain_hardening_indexs_test)))
ns = np.zeros((len(yield_strengths_test), len(strain_hardening_indexs_test)))


scratch_depth = 50e-3
indenter_angle = 120
indenter_radius_at_scratch_depth = scratch_depth / np.tan(
    (90 - indenter_angle / 2) * np.pi / 180
)

for i, sy in enumerate(yield_strengths_test):
    count = 0
    for j, n in enumerate(strain_hardening_indexs_test):
        # print(yield_strengths)
        fileNameID = (
            "SY"
            + str(sy)
            + "_n0"
            + str(int(10 * n))
            + "_d0050_E200000_mu00_rho78_Poisson03"
        )
        path = "ScratchData/TestData/"
        rfs, coords = LoadScratchTestData(
            fileNameID=fileNameID, path=path, toLoad="both"
        )
        # x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        h_r, w, h_p = GetTopographyData(coords, 2.00, 2.44)
        h_rs[i, j] = h_r
        ws[i, j] = w
        h_ps[i, j] = h_p
        rf2_test[i, j] = np.mean(rfs[-4 - 20 : -4, 2])
        rf3_test[i, j] = np.mean(rfs[-4 - 20 : -4, 3])
        rf2[i, j] = abs(rfs[-4, 2])
        rf3[i, j] = rfs[-4, 3]
        sy_s[i, j] = sy
        ns[i, j] = round(n, 1)

H_s = abs(rf2) / (1 / 4 * (ws) ** 2)

# %%
rfs, coords = LoadScratchTestData(fileNameID=fileNameID, path=path, toLoad="both")
plt.figure()
plt.plot(abs(rfs[:, 3]))

# %% plot F_t/F_n vs sy
plt.figure()

# plt.plot(yield_strengths, abs(rf2), c="b")
# plt.plot(yield_strengths, rf3, c="k")
color = ["b", "r", "k"]
for i in range(3):
    plt.plot(
        yield_strengths_test / youngs_modulus,
        abs(rf3_test[:, i]) / abs(rf2_test[:, i]),
        "o-",
        color=color[i],
        markersize=5,
        label=f"Averge measure, n={ns[1,i]}",
    )
    plt.plot(
        yield_strengths_test / youngs_modulus,
        rf3[:, i] / abs(rf2[:, i]),
        "x-",
        color=color[i],
        markersize=5,
        label="last point measure",
    )
plt.legend()
plt.xlabel(r"$\sigma_y/E$ [-]")
plt.ylabel(r"$F_t/F_n$ [-]")
plt.grid()
plt.ylim([0, 0.4])
# plt.xlim([0, 0.01])

# %% Plotting forces over simulation time
path = "ScratchData/"
fileNameID1 = "ExplicitRigidIndenterScratch"
rfs, coords = LoadScratchTestData(fileNameID=fileNameID1, path=path, toLoad="both")
idx = np.where(rfs[:, 0] == 0)[0]
time = rfs[:, 0]
time[idx[1] :] = rfs[idx[1] :, 0] + rfs[idx[1] - 1, 0]
time[idx[2] :] = rfs[idx[2] :, 0] + rfs[idx[2] - 1, 0]
z = np.linspace(0, 2, len(time[idx[1] : idx[2]]))
N = 5
rf2_movingMean = np.convolve(abs(rfs[idx[1] : idx[2], 2]), np.ones(N) / N, mode="valid")
z_movingMean = np.linspace(0, 2, len(rf2_movingMean))
plt.plot(z, abs(rfs[idx[1] : idx[2], 2]))
plt.plot(z_movingMean, rf2_movingMean)
plt.vlines([2.00 - 0.44, 2.00], 0, 100, "k", "--", label="Zone of data extraction")
plt.ylim([40, 100])
plt.grid()
