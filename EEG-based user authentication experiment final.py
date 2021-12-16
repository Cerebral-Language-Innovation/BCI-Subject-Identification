# may 7th 2021
# 15 subs 6 ica_comp. each
# combs = task 1 + task 3, task 2 + task 4
# total = 3runs*2combs = 6 data points for ea. sub
# the reason why only 6: a person never has the exact same brain data,
# so we can't reuse a run. 12 runs total => max 6 pts per sub if we want to have
# a sequence of movements as the data pts.

import os
#import Mlp (just combined them)
import keras
import mne
from mne.preprocessing import ICA, create_ecg_epochs
from mne import pick_types
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.metrics
import numpy as np
from collections import Counter
import math
from sklearn.utils import resample
from keras import metrics
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve

# 1. get ica, make runs_set
runs_set = []

task1 = [3,7,11]
task2 = [4,8,12]
task3 = [5,9,13]
task4 = [6,10,14]

# the ica function
def getIca(subject, runs):
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)
    raw.rename_channels(lambda x: x.strip('.'))
    raw.filter(14., 30.)
    picks = pick_types(raw.info, eeg=True)
    ica = ICA(n_components=15, random_state=97)
    ica.fit(raw)
    raw.load_data()
    icaArray = ica.get_components()
    return icaArray


# create the runs_set
for i in range(1,108):
    hand_set = []
    feet_set = []
    for r in range(3):
        runA = getIca(i, task1[r]) #ica for run in task1
        runB = getIca(i, task3[r]) #ica for run in task3
        hand = np.concatenate((runA,runB),axis=1) # person's task 1 then 3
        runC = getIca(i, task2[r])
        runD = getIca(i, task4[r])
        feet = np.concatenate((runC,runD),axis=1)
        hand_set.append(hand)
        feet_set.append(feet)

    for i in range(15):
        hand = hand_set[np.random.randint(0,3)]
        feet = feet_set[np.random.randint(0,3)]
        hand2 = hand_set[np.random.randint(0,3)]
        runs_set.append(np.concatenate((hand,feet,hand2),axis=1))



# 2. SMOTE resample to have 1:1
data = runs_set
# the smote function
def smote_resample(x,y):
    smote = SMOTE(random_state=42, sampling_strategy='minority')
    sx, sy = smote.fit_resample(x, y)
    return sx, sy

# get x and y
x,y = [],[]
for i in range(len(data)):
    features = np.array([feature for feature in data[i][0]]).flatten()
    x.append(features)
    if i<15:
        y.append(1)
    else:
        y.append(0)
print(y)

# use smote
sx, sy = smote_resample(x,y)

# 3. Classification
print(Counter(sy))
x_train, x_test, y_train, y_test = train_test_split(sx, sy, test_size=0.3)
x_train = normalize(np.asarray(x_train))
x_test = normalize(np.asarray(x_test))
print(x_train)
y_train = np.asarray(y_train).astype('int32').reshape((-1, 1))
y_test = np.asarray(y_test).astype('int32').reshape((-1, 1))
from tensorflow.keras.models import Sequential
model = Sequential([
   Dense(30, activation='relu'),
   Dropout(0.75),
   Dense(15, activation='relu'),
   Dropout(0.375),
   Dense(10, activation='relu'),
   Dense(1, activation='sigmoid')])

model.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['accuracy','TruePositives','TrueNegatives','FalsePositives','FalseNegatives'])
model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test), verbose=2)
y_pred = model.predict_classes(x_test)
