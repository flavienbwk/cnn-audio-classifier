import matplotlib
matplotlib.use('Agg')

import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from keras.models import Sequential
import pylab
import librosa
import librosa.display
import numpy as np
import pandas as pd
import cv2
import random
import time
import math

import warnings
warnings.filterwarnings('ignore')

# Read Data
data = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
data.head(5)

# Get data over 3 seconds long
valid_data = data[['slice_file_name', 'fold' ,'classID', 'class']][data['end'] - data['start'] >= 3]
valid_data.shape

# Example of a Siren spectrogram
y, sr = librosa.load('UrbanSound8K/audio/fold6/135160-8-0-0.wav', duration=2.97)
ps = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(ps, y_axis='mel', x_axis='time')
cv2.imwrite('spectre1.png', cv2.flip(ps, -1))

# Example of a AC spectrogram
y, sr = librosa.load('UrbanSound8K/audio/fold1/134717-0-0-19.wav', duration=2.97)
ps = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(ps, y_axis='mel', x_axis='time')
cv2.imwrite('spectre2.png', cv2.flip(ps, -1))

# Example of a children playing spectrogram
y, sr = librosa.load('UrbanSound8K/audio/fold9/13579-2-0-16.wav', duration=2.97)
ps = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(ps, y_axis='mel', x_axis='time')
cv2.imwrite('spectre3.png', cv2.flip(ps, -1))

y, sr = librosa.load('UrbanSound8K/audio/fold9/137815-4-0-0.wav', duration=2.97)
ps = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(ps, y_axis='mel', x_axis='time')
cv2.imwrite('spectre4.png', cv2.flip(ps, -1))

valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')


D = [] # Dataset
print "Found " + str(len(valid_data)) + " samples..."
time.sleep(2)

limit = 8000
print "Limiting load at " + str(limit) + " samples."

count = 1
count_loaded = 0
for row in valid_data.itertuples():
    if count > limit: break
    print "Loading " + row.path + " (" + str(count) + ")"
    try:
        y, sr = librosa.load('UrbanSound8K/audio/' + row.path, duration=2.97)  
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        if ps.shape != (128, 128): continue
        D.append( (ps, row.classID) )
        count_loaded += 1
    except:
        print "Failed to load."
    count += 1

print("Number of samples: ", len(D))


dataset = D
random.shuffle(dataset)

selection = int(math.floor(count_loaded * 0.9))
print "Samples from 1 to " + str(selection) + " will be selected as training samples."
print "Samples from " + str(selection + 1) + " to " + str(count_loaded) + " will be selected as test samples."
train = dataset[:selection]
test = dataset[selection + 1:]

X_train, y_train = zip(*train)
X_test, y_test = zip(*test)

# Reshape for CNN input
X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
X_test = np.array([x.reshape( (128, 128, 1) ) for x in X_test])

# One-Hot encoding for classes
y_train = np.array(keras.utils.to_categorical(y_train, 10))
y_test = np.array(keras.utils.to_categorical(y_test, 10))

model = Sequential()
input_shape=(128, 128, 1)

model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(
	optimizer="Adam",
	loss="categorical_crossentropy",
	metrics=['accuracy'])

model.fit(
	x=X_train, 
	y=y_train,
    epochs=12,
    batch_size=128,
    validation_data= (X_test, y_test))

score = model.evaluate(
	x=X_test,
	y=y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
