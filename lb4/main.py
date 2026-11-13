import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DATA = "data-apty.csv"

BETA0_LIST = [
    # A   B     T       phi
    [0.0, 5.0,  365.25, 0.0],
    [0.0, 5.0,  90.0,   0.0],
    [0.0, 10.0, 180.0,  1.0],
]

def modelfun(t, A, B, T, phi):
    return A + B * np.sin(2 * np.pi * t / T + phi)

df = pd.read_csv(DATA, sep=";")
df["date"] = pd.to_datetime(df["date"])
t = (df["date"] - df["date"].min()).dt.days.to_numpy(dtype=float) + 1.0
y = df["value"].to_numpy(dtype=float)
N = len(y)
ss_tot = float(np.sum((y - y.mean()) ** 2))

fits = []
best = None
best_i = 0
errs = []
for i, b0 in enumerate(BETA0_LIST):
    beta, _ = curve_fit(modelfun, t, y, p0=b0, maxfev=20000)
    err = y - modelfun(t, *beta)
    errs.append(err)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = 1.0 - float(np.sum(err ** 2)) / ss_tot
    fits.append((beta, rmse, r2))
    if best is None or rmse < best[1]:
        best = (beta, rmse, r2)
        best_i = i

    print(f"beta0[{i}] = {b0}")
    print(f"beta[{i}] = [{', '.join(f'{b:.4f}' for b in beta)}]")
    print(f"rmse = {rmse:.4f}")
    print(f"R^2  = {r2:.4f}")
    print()

def harmonics(t, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10):
    return (B1 *  np.cos(2 * np.pi * 1 * t / 730.5)
          + B2 *  np.sin(2 * np.pi * 1 * t / 730.5)
          + B3 *  np.cos(2 * np.pi * 2 * t / 730.5)
          + B4 *  np.sin(2 * np.pi * 3 * t / 730.5)
          + B5 *  np.cos(2 * np.pi * 4 * t / 730.5)
          + B6 *  np.sin(2 * np.pi * 4 * t / 730.5)
          + B7 *  np.cos(2 * np.pi * 8 * t / 730.5)
          + B8 *  np.sin(2 * np.pi * 8 * t / 730.5)
          + B9 *  np.cos(2 * np.pi * 16 * t / 730.5)
          + B10 * np.sin(2 * np.pi * 16 * t / 730.5))

p0 = [0.0] * 10
c_h, _ = curve_fit(harmonics, t, y, p0=p0, maxfev=20000)
err_h = y - harmonics(t, *c_h)
rmse_h = float(np.sqrt(np.mean(err_h ** 2)))
r2_h = 1.0 - float(np.sum(err_h ** 2)) / ss_tot

print(f"B1 = {c_h[0]:.4f}, B2  = {c_h[1]:.4f}")
print(f"B3 = {c_h[2]:.4f}, B4  = {c_h[3]:.4f}")
print(f"B5 = {c_h[4]:.4f}, B6  = {c_h[5]:.4f}")
print(f"B7 = {c_h[6]:.4f}, B8  = {c_h[7]:.4f}")
print(f"B9 = {c_h[8]:.4f}, B10 = {c_h[9]:.4f}")
print(f"RMSE = {rmse_h:.4f}")
print(f"R^2 = {r2_h:.4f}")

t_grid = np.linspace(t.min(), t.max(), 1000)
y_h_grid = harmonics(t_grid, *c_h)

for i, (betai, rmsi, r2i) in enumerate(fits):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, y, ".", ms=4, color="black", label="Исходные данные")
    ax.plot(t_grid, modelfun(t_grid, *betai), color="red", label="Предсказание модели")
    ax.set_xlabel("День")
    ax.set_ylabel("RCORR_E")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"img/fit_start_{i}.png", dpi=150)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, y, ".", ms=4, color="black", label="Исходные данные")
ax.plot(t_grid, y_h_grid, color="green", label="Предсказание модели")
ax.set_xlabel("День")
ax.set_ylabel("RCORR_E")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("img/harmonics.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, y, ".", ms=4, color="black", label="data")
for i, (betai, rmsi, r2i) in enumerate(fits):
    if i == best_i: continue
    ax.plot(t_grid, modelfun(t_grid, *betai), color="gray", lw=1, label=f"Предсказание модели {i+1}")
ax.plot(t_grid, modelfun(t_grid, *fits[0][0]), color="red", lw=2,
        label=f"Предсказание лучшей модели")
ax.set_xlabel("День")
ax.set_ylabel("RCORR_E")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("img/compare_all.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(errs[0], bins=25, color="gray", edgecolor="black")
ax.set_xlabel("Ошибка")
ax.set_ylabel("Дни")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("img/errors_hist.png", dpi=150)
plt.close(fig)