# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import csv
import numpy as np
import keras.utils
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.optimizers

outputs = []
inputs = []

outputsValid = []
inputsValid = []

inputsPredict = []

with open('Data.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        outputs.append(row[7])
        inputs.append(row[:7])
        inputs[-1] = [int(elem) for elem in inputs[-1]]

with open('DataValid.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        outputsValid.append(row[7])
        inputsValid.append(row[:7])
        inputsValid[-1] = [int(elem) for elem in inputsValid[-1]]

with open('DataPredict.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        inputsPredict.append(row[:7])
        inputsPredict[-1] = [int(elem) for elem in inputsPredict[-1]]


inputs = np.asarray(inputs)
outputs = np.asarray(outputs)
inputs = np.reshape(inputs, (len(inputs),1,7))
outputs = keras.utils.to_categorical(outputs)
outputs = np.reshape(outputs, (len(outputs),1,10))

inputsValid = np.asarray(inputsValid)
outputsValid = np.asarray(outputsValid)
inputsValid = np.reshape(inputsValid, (len(inputsValid),1,7))
outputsValid = keras.utils.to_categorical(outputsValid)
outputsValid = np.reshape(outputsValid, (len(outputsValid),1,10))

inputsPredict = np.asarray(inputsPredict)
inputsPredict = np.reshape(inputsPredict, (len(inputsPredict),1,7))


sgd = keras.optimizers.SGD(lr=0.05)

model = Sequential()
model.add(Dense(16, input_dim=7, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(10, activation='softmax'))

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

model.summary()
model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(inputs, outputs, callbacks=[callback], epochs = 500, validation_data=(inputsValid, outputsValid))

prediction = model.predict(inputsPredict)
prediction =  np.reshape(prediction, (len(prediction),10))
print([np.argmax(elem) for elem in prediction])

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
