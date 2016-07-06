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
print('Start')
df_train = pd.read_csv('../input/train.csv')
df_test= pd.read_csv('../input/test.csv')
for i in range(0,784):
    df_train.loc[df_train['pixel' + str(i)] < 80, 'pixel' + str(i)] = 0
    df_train.loc[df_train['pixel' + str(i)] >= 80, 'pixel' + str(i)] = 1
    df_test.loc[df_test['pixel' + str(i)] < 80, 'pixel' + str(i)] = 0
    df_test.loc[df_test['pixel' + str(i)] >= 80, 'pixel' + str(i)] = 1

y_train = df_train['label']
X_train = df_train.ix[:,1:]

print('Start make model')
clf = KNeighborsClassifier(n_neighbors=3,
                           weights=calculate_distance, p=1,
                           n_jobs=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(df_test)

print('Make Submission file')
sub_file = os.path.join('../Submission/submission.csv')
out = open(sub_file, "w")
out.write("ImageId,Label\n")

row = 1
for val in y_pred:
    print(str(row) + ',' + str(val))
    out.write(str(row) + ',' + str(val) + '\n')
    row += 1
out.close()

elapsed = (time.time() - start_time)
print('Task completed in:', datetime.timedelta(seconds=elapsed))
os.system('say "Tom tom where you go last night , I love maung thai , I like pat pong"')