#
# This file simulates a sequential training of
# a random forest grid search in a manner similar
# to what is implemented in src/
#

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Whether or not to run the grid search in parallel
# on the CPU
PARALLEL = True

print("Loading Data...")
f = "test_data/xy.csv"
buffersize = 79739
d = pd.read_csv(f)
while len(d) < buffersize:
    d = d.append(pd.read_csv(f))
    print(len(d) / buffersize)
out = d.iloc[:buffersize,:]

x = out.iloc[:, :-1].values
y = out.iloc[:, -1].astype(int).values


print('X:\n', x)
print('Y:\n', y)
print("Done.")

print("Training RandomForestClassifier")

p = {
        'n_estimators': [10, 100, 1000],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 100]
    }

rf = RandomForestClassifier(verbose=True)

if PARALLEL:
    clf = GridSearchCV(rf, p, n_jobs=-1)
else:
    clf = GridSearchCV(rf, p, n_jobs=1)

clf.fit(x, y)

print('Done.')
