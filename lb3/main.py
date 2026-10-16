import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pathlib import Path

#VALUE_COLUMN="RCORR_E"
VALUE_COLUMN="value"
DATA="data_raw"
SEED = 123
FEATURE_NAMES = [
    "mean", "std", "amp", "median",
    "min", "max",
    "p25", "p75", "p01", "p99",
]
M=2

def load_day(path: Path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    full = pd.date_range(df.index.min().normalize(), periods=1440, freq="1min")
    df = df.reindex(full)
    median = float(df[VALUE_COLUMN].median())
    df[VALUE_COLUMN] = df[VALUE_COLUMN].fillna(median)
    return df[VALUE_COLUMN].to_numpy(dtype=float)

def features(s):
    m = float(s.mean())
    sd = float(s.std())
    med = float(np.median(s))
    p25, p75 = np.percentile(s, [25, 75])
    p01, p99 = np.percentile(s, [1, 99])
    return np.array([
        m, sd, s.max() - s.min(), med,
        s.min(), s.max(),
        p25, p75, p01, p99,
    ])

def load_all():
    X, y = [], []
    for cls in (0, 1):
        for path in sorted((Path(DATA) / f"class{cls}").glob("*.csv")):
            series = load_day(path)
            X.append(features(series))
            y.append(cls)
    return np.array(X), np.array(y)

def match_to_pred_labels(y_true, y_pred):
    acc1 = float(np.mean(y_pred == y_true))
    acc2 = float(np.mean(y_pred == 1 - y_true))
    if acc2 > acc1:
        return acc2, 1 - y_pred
    return acc1, y_pred

def confusion(y_true, y_pred):
    return np.array([[np.sum((y_true == 0) & (y_pred == 0)),
                      np.sum((y_true == 0) & (y_pred == 1))],
                     [np.sum((y_true == 1) & (y_pred == 0)),
                      np.sum((y_true == 1) & (y_pred == 1))]])

def fcm(X, n_clusters, m, max_iter, tol, seed):
    n = len(X)
    U = np.random.default_rng(seed).random((n, n_clusters))
    U = U / U.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        w = U ** m
        centers = (w.T @ X) / w.sum(axis=0)[:, None]

        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        d2 = np.maximum(d2, 1e-12)
        U_new = 1.0 / d2 ** (1.0 / (m - 1))
        U_new = U_new / U_new.sum(axis=1, keepdims=True)

        if np.abs(U_new - U).max() < tol:
            U = U_new
            break
        U = U_new

    w = U ** m
    centers = (w.T @ X) / w.sum(axis=0)[:, None]
    return centers, U

def draw_gist(ys, filename):
    cols = {0: "#FF0000", 1: "#0000FF"}
    fig, axes = plt.subplots(4, 3, figsize=(10, 7))
    for i, ax in enumerate(axes.flat):
        if i >= len(FEATURE_NAMES):
            ax.axis("off")
            continue

        f0, f1 = X[ys == 0, i], X[ys == 1, i]
        bins = np.histogram_bin_edges(np.concatenate([f0, f1]), bins=20)
        ax.hist(f0, bins=bins, alpha=0.6, color=cols[0])
        ax.hist(f1, bins=bins, alpha=0.6, color=cols[1])
        ax.set_title(FEATURE_NAMES[i])

    fig.legend(handles=[Patch(color=cols[0], label="магнитные бури"),
                        Patch(color=cols[1], label="спокойные дни")],
            loc="lower center")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

X, y = load_all()

std = X.std(axis=0)
std[std == 0] = 1.0
Xn = (X - X.mean(axis=0)) / std

centers, U = fcm(Xn, 2, M, 100, 1e-4, SEED)
fpc = float(np.mean(np.sum(U ** 2, axis=1)))

y_pred = U.argmax(axis=1)
acc, y_pred = match_to_pred_labels(y, y_pred)
cm = confusion(y, y_pred)

print(f"accuracy: {acc:.3f}")
print("confusion matrix:")
print(cm)

tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
recall = tp / (tp + fn)
precision = tp / (tp + fp)
print(f"recall (спокойные): {recall:.3f}")
print(f"precision (спокойные): {precision:.3f}")
print(f"FPC: {fpc:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
for ax, colors, title in [
    (axes[0], y, "Истинные классы"),
    (axes[1], y_pred, "Кластеры FCM"),
]:
    sc = ax.scatter(X[:, 0], X[:, 1], c=colors, cmap="viridis", s=20, alpha=0.5)
    ax.set_xlabel(FEATURE_NAMES[0])
    ax.set_ylabel(FEATURE_NAMES[1])
    ax.set_title(title)
    ax.grid(alpha=0.3)

fig.legend(handles=[Patch(color=plt.cm.viridis(0), label="класс/кластер 0 (спокойный)"),
                    Patch(color=plt.cm.viridis(0.99), label="класс/кластер 1 (буря)")],
           loc="upper center", ncol=2, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("img/fcm_clusters.png", dpi=150)

draw_gist(y, "img/feat_hists.png")
draw_gist(y_pred, "img/feat_hists_pred.png")