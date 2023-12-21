#%%
# Load all necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from keras.utils import to_categorical
from keras.applications import EfficientNetB0
from keras.optimizers import Adam
from keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import os

#%%
# Load data
X = []
y = []
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
absolute_path = os.path.dirname(__file__)

for i in labels:
    datapath = os.path.join(absolute_path, 'Data/', i)
    for j in os.listdir(datapath):
        mri = cv2.imread(os.path.join(datapath, j))
        mri = cv2.resize(mri, (150, 150))
        X.append(mri)
        y.append(i)

# Convert to arrays
X = np.array(X)
y = np.array(y)

#%%
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# One-hot encode the data
y_train_OH = []
for i in y_train:
    y_train_OH.append(labels.index(i))
y_train = y_train_OH
y_train = to_categorical(y_train)


y_test_OH = []
for i in y_test:
    y_test_OH.append(labels.index(i))
y_test = y_test_OH
y_test = to_categorical(y_test)

#%%
# Load the EfficientNetB0 model pre-trained on ImageNet data
effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Build a custom model on top of EfficientNetB0
base_model = effnet.output
base_model = AveragePooling2D(pool_size = (4,4))(base_model)
base_model = Flatten(name= 'flatten')(base_model)
base_model = Dense(256, activation = "relu")(base_model)
base_model = Dropout(0.25)(base_model)
base_model = Dense(256, activation = "relu")(base_model)
base_model = Dropout(0.25)(base_model)
base_model = Dense(256, activation = "relu")(base_model)
base_model = Dropout(0.25)(base_model)
base_model = Dense(4, activation = 'softmax')(base_model)

# Freeze the convolutional layers
for layer in effnet.layers:
    layer.trainable = False

# Compile the model
model = Model(inputs = effnet.input, outputs = base_model)
model.compile(optimizer = Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#%%
# Define callbacks
checkpoint = ModelCheckpoint('effnet_model.keras', save_best_only=True, monitor='val_accuracy', mode='auto', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)

# Train the model,
history = model.fit(X_train, y_train, validation_split = 0.1, epochs = 12, verbose = 1, batch_size = 32, callbacks = [checkpoint, reduce_lr])

#%%
# Plot training history
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Gather predictions
y_pred = model.predict(X_test, batch_size = 32)
pred = np.argmax(y_pred, axis = 1)
y_test_new = np.argmax(y_test, axis=1)

# Error metrics
print('Validation Accuracy:', max(history.history['val_accuracy']))
print('MSE:', mean_squared_error(y_test_new, pred))
print('R2:', r2_score(y_test_new, pred))

# Classification report
print(classification_report(y_test_new, pred))
# %%
