import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import skfuzzy
from pathlib import Path

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
    median = float(df["RCORR_E"].median())
    df["RCORR_E"] = df["RCORR_E"].fillna(median)
    return df["RCORR_E"].to_numpy(dtype=float)

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
        for path in sorted((Path("data") / f"class{cls}").glob("*.csv")):
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

cntr, U, _, _, _, p, fpc = skfuzzy.cmeans(Xn.T, c=2, m=M, error=1e-4, maxiter=100, seed=SEED)
print(f"converged in {p} iters")

y_pred = U.argmax(axis=0)
acc, y_pred = match_to_pred_labels(y, y_pred)
cm = confusion(y, y_pred)

print(f"accuracy: {acc:.3f}")
print("confusion matrix:")
print(cm)

tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
recall = tp / (tp + fn)
precision = tp / (tp + fp)
print(f"recall (бури): {recall:.3f}")
print(f"precision (бури): {precision:.3f}")
print(f"FPC: {fpc:.3f}")

draw_gist(y, "img/feat_hists.png")
draw_gist(y_pred, "img/feat_hists_pred.png")