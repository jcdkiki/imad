import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv', skiprows=25, sep=';', names=['datetime', 'RCORR_E'])
df['datetime'] = pd.to_datetime(df['datetime'])
df['RCORR_E'] = pd.to_numeric(df['RCORR_E'])

print(df.head())

mean = df['RCORR_E'].mean()
std  = df['RCORR_E'].std()

print(f"mean: {mean}")
print(f"std: {std}")

range = (float(mean - 3*std), float(mean + 3*std))
print(f"range: {range}")

n_total = len(df)

mask_outliers = (df['RCORR_E'] < range[0]) | (df['RCORR_E'] > range[1])
n_outliers = int(mask_outliers.sum())

print(f"total: {n_total}")
print(f"outliers: {n_outliers} ({n_outliers/14400*100:.3f}%)")

df_clean = df['RCORR_E'].drop(df[mask_outliers].index)
median = df_clean.median()
print(f"median: {median}")

df_new = df.copy()
df_new.loc[mask_outliers, 'RCORR_E'] = median

df_diff = df.datetime.diff()

print(df_diff[df_diff > pd.Timedelta('1min')])

n_missing = 14400 - len(df['RCORR_E'])
print(f"n_missing: {n_missing} ({n_missing/14400*100:.3f}%)")

print(f"n_missing+outliers: {n_missing+n_outliers} ({(n_missing+n_outliers)/n_total*100:.3f}%)")

df_new = df.set_index('datetime').reindex(pd.date_range(df.datetime.min(), df.datetime.max(), freq='1min')).reset_index()
df_new.columns = ['datetime', 'RCORR_E']
df_new['RCORR_E'] = df_new['RCORR_E'].fillna(median)

assert(len(df_new['RCORR_E']) == 14400)

minx = df.index.min()
maxx = df.index.max()

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
ax.plot(df.index, df['RCORR_E'])
ax.set_xlabel('time')
ax.set_ylabel('RCORR_E')
ax.set_title('Raw data')
ax.plot((minx, maxx), (range[0], range[0]), 'r', linewidth=0.5, label="3 sigma range")
ax.plot((minx, maxx), (range[1], range[1]), 'r', linewidth=0.5)
ax.legend(loc=1)
fig.savefig("img/raw.png")

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
ax.plot(df_new.index, df_new['RCORR_E'],)
ax.set_xlabel('time')
ax.set_ylabel('RCORR_E')
ax.set_title('Processed data')
ax.plot((minx, maxx), (range[0], range[0]), 'r', linewidth=0.5, label="3 sigma range")
ax.plot((minx, maxx), (range[1], range[1]), 'r', linewidth=0.5)
ax.plot((minx, maxx), (median, median), color='lime', linewidth=1, label="median")
ax.legend(loc=1)
fig.savefig("img/processed.png")

print(df['RCORR_E'].min())
print(df['RCORR_E'].max())