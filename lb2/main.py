import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

M = 180
K_VALUES = [2, 3, 4]

def euclidean(a, b):
    d = a - b
    return np.sqrt((d * d).sum())

def manhattan(a, b):
    return np.abs(a - b).sum()

def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 - (a @ b.T) / denom

DIST_FUNCS = [
    ("euclidean", euclidean),
    ("manhattan", manhattan),
    ("cosine", cosine)
]

df = pd.read_csv("data.csv", skiprows=25, sep=";", names=["datetime", "RCORR_E"])
df["datetime"] = pd.to_datetime(df["datetime"])
df["RCORR_E"] = pd.to_numeric(df["RCORR_E"])

mean = df["RCORR_E"].mean()
std = df["RCORR_E"].std()
range_min = float(mean - 3 * std)
range_max = float(mean + 3 * std)
mask_outliers = (df["RCORR_E"] < range_min) | (df["RCORR_E"] > range_max)

df_clean = df["RCORR_E"].drop(df[mask_outliers].index)
median = df_clean.median()

df.loc[mask_outliers, "RCORR_E"] = median
full = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="1min")
df = df.set_index("datetime").reindex(full)
df["RCORR_E"] = df["RCORR_E"].fillna(median)

series = df["RCORR_E"].to_numpy(dtype=float)
N = len(series)

history_vecs = [series[i:i + M] for i in range(N - M + 1)]

predict_targets = range(N - M - M, N - M)

true_vals = series[N - M:]
preds = {}

for norm_name, norm in DIST_FUNCS:
    print(f"norm={norm_name}")
    preds[norm_name] = {k: [] for k in K_VALUES}
    for j in tqdm.tqdm(predict_targets):
        query = history_vecs[j]
        prev_history_vecs = history_vecs[:j]
        dists = np.array([norm(query, w) for w in prev_history_vecs])
        order = np.argsort(dists)
        for k in K_VALUES:
            nb = order[:k]
            preds[norm_name][k].append(series[nb + M].mean())

results = {}
for norm_name, _ in DIST_FUNCS:
    for k in K_VALUES:
        p = np.array(preds[norm_name][k])
        err = p - true_vals
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        corr = float(np.corrcoef(p, true_vals)[0, 1])
        results[(norm_name, k)] = (mae, rmse, corr, p)

        print(f"{norm_name} k={k}: MAE={mae:.3f}, RMSE={rmse:.3f}, corr={corr:+.3f}")

        fig, ax = plt.subplots(figsize=(12, 4))
        t = np.arange(N - M, N)
        ax.plot(t, true_vals, label="истина", lw=1.0, c="black")
        ax.plot(t, p, label=f"прогноз ({norm_name}, k={k})", c="red")
        ax.set_xlabel("Отсчёт");
        ax.set_ylabel("RCORR_E")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"img/predict_{norm_name}_k{k}.png")