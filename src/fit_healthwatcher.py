import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import os

DATA_DIR = '../data/0430/community'

gts = pd.read_pickle(os.path.join(DATA_DIR, 'ground_truths_sample'))
meta = pd.read_pickle(os.path.join(DATA_DIR, 'meta'))
df = pd.read_pickle(os.path.join(DATA_DIR, 'sample_data'))

df[['mean_red', 'std_red', 'mean_green', 'std_green', 'mean_blue', 'std_blue']] = df[['mean_red', 'std_red', 'mean_green', 'std_green', 'mean_blue', 'std_blue']].apply(pd.to_numeric)

df = df.groupby(['sample_id']).mean()

y = gts['SpO2'].to_numpy().reshape(-1,1)

mean_red = df['mean_red'].to_numpy()
std_red = df['std_red'].to_numpy()
mean_blue = df['mean_blue'].to_numpy()
std_blue = df['std_blue'].to_numpy()
mean_green = df['mean_green'].to_numpy()
std_green = df['std_green'].to_numpy()

X = np.array([mean_red, std_red, mean_blue, std_blue, mean_green]).T # <-- 'new' method, linear regression on all features
# X = (std_red / mean_red) / (std_blue / mean_blue).reshape(-1,1) <-- healthwatcher's method, aggregates all into 1

split=int(len(X)*0.75)

X_train=X[:split]
X_test=X[split:]

y_train=y[:split]
y_test=y[split:]

reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X, y))
print(reg.coef_)
print(reg.intercept_)

y_pred = reg.predict(X_test)

import matplotlib.pyplot as plt

plt.scatter(x=y_pred, y=y_test)
plt.xlabel('predictions')
plt.ylabel('ground truth values')
plt.show()
import time

pd.DataFrame(data={'spo2_gt': y_test.reshape(1,-1)[0], 'spo2_pred': y_pred.reshape(1,-1)[0]}).to_csv(f'../results/hw-fit-{int(time.time())}')



