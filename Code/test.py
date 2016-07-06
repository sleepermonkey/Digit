import numpy as np
import pandas as pd
import time
import datetime
import os
import operator
from sklearn.neighbors import KNeighborsClassifier

def calculate_distance(distances):
    return distances ** -2

start_time = time.time()
df = pd.read_csv('../input/train.csv')
for i in range(0,784):
    df.loc[df['pixel' + str(i)] < 80,'pixel' + str(i)] = 0
    df.loc[df['pixel' + str(i)] >= 80,'pixel' + str(i)] = 1

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

y_train = train['label']
X_train = train.ix[:,1:]

y_test = test['label']
X_test = test.ix[:,1:]

y_pred = np.zeros(shape=y_test.shape)

clf = KNeighborsClassifier(n_neighbors=3,
                           weights=calculate_distance, p=1,
                           n_jobs=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

Score = 0
Total = 0
for t, p in zip(y_test, y_pred):
    if t == p:
        Score += 1
    else:
        print(str(t) + ',' + str(p))
    Total += 1

print(float(Score) / float(Total))
elapsed = (time.time() - start_time)
print('Task completed in:', datetime.timedelta(seconds=elapsed))
os.system('say "Tom tom where you go last night , I love maung thai , I like pat pong"')