# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:29:17 2020

@author: georg
"""

import random
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

pictures = []
labels = []
planesnet_dir = ('planesnet')

#%% load images
files = glob.glob (planesnet_dir + "/*.png") # image path
# shuffle
random.Random(42).shuffle(files)

for picture in files:
    image = cv2.imread (picture)
    im_name = picture.replace(planesnet_dir + "\\","")
    labels.append(im_name[0])
    pictures.append(image)

plt.imshow(pictures[0])
plt.title(labels[0])

#%%
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(pictures[i])
    plt.title(labels[i])
plt.show()

#%%
develop = False
# 32,000 Instanzen - 20,480 Train, 5,120 Val, 6400 Test

if develop:
    X_train = np.array(pictures[0:5000]).astype('float32')
    X_train /= 255
    y_train = np.array(labels[0:5000])
    X_val = np.array(pictures[5000:6000]).astype('float32')
    X_val /= 255
    y_val = np.array(labels[5000:6000])
    X_test = np.array(pictures[6000:7000]).astype('float32')
    X_test /= 255
    y_test = np.array(labels[6000:7000])
else:
    X_train = np.array(pictures[0:20480]).astype('float32')
    X_train /= 255
    y_train = np.array(labels[0:20480])
    X_val = np.array(pictures[20480:25600]).astype('float32')
    X_val /= 255
    y_val = np.array(labels[20480:25600])
    X_test = np.array(pictures[25600:]).astype('float32')
    X_test /= 255
    y_test = np.array(labels[25600:])

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

epochs = 50
batch_size = 128


model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(20,20,3)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

#%%
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#%%
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
y_pred_val = model.predict(X_val)
y_pred_val = np.argmax(y_pred_val, axis=1)
y_pred_test = model.predict(X_test)
y_pred_test = np.argmax(y_pred_test, axis=1)
y_val = np.argmax(y_val, axis=1)
y_test = np.argmax(y_test, axis=1)
print(" ### Validation Set ### ")
print("confusion matrix:")
print(confusion_matrix(y_val, y_pred_val))
print("")
print("Accuracy: ", accuracy_score(y_val, y_pred_val))
print('\n')
print(classification_report(y_val, y_pred_val, target_names=["no plane","plane"]))


print("### Test Set ### ")
print("confusion matrix:")
print(confusion_matrix(y_test, y_pred_test))
print("")
print("Accuracy: ", accuracy_score(y_test, y_pred_test))
print('\n')
print(classification_report(y_test, y_pred_test, target_names=["no plane","plane"]))