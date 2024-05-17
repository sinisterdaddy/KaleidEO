import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau 
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import cv2
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Define UNet architecture functions
def conv_bl(inp, filters, dropout, activation="relu", kernel_initializer="he_normal", trainable=False):
    con1 = Conv2D(filters, (3,3), activation=activation, kernel_initializer=kernel_initializer, padding="same")(inp)
    d1 = Dropout(dropout)(con1)
    con1 = Conv2D(filters, (3,3), activation=activation, kernel_initializer=kernel_initializer, padding="same", trainable=trainable)(d1)
    return con1

def enc(inp, filters, dropout, activation="relu", kernel_initializer="he_normal", trainable=False):
    c1 = conv_bl(inp, filters, dropout, activation=activation, kernel_initializer=kernel_initializer)
    c2 = conv_bl(c1, filters, dropout, activation=activation, kernel_initializer=kernel_initializer, trainable=trainable)
    p1 = MaxPooling2D((2,2))(c2)
    return c2, p1

def dec(inp, skip, filters, dropout, activation="relu", kernel_initializer="he_normal", trainable=False):
    up1 = Conv2DTranspose(filters, (2,2), (2,2))(inp)
    s1 = concatenate([up1, skip])
    c1 = conv_bl(s1, filters, dropout, activation=activation, kernel_initializer=kernel_initializer, trainable=trainable)
    return c1

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3]) + K.sum(y_pred,[1,2,3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def resize_images(images):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, (256, 256))
        resized_images.append(resized_img)
    return np.array(resized_images)

# Define model constants
IMG_HEIGHT = 256
IMG_WIDTH = 256
BANDS = 3
inp_path = "" 
model_path = ""

# Load and preprocess dataset
with open('C:/Users/krish/OneDrive/Documents/GitHub/KaleidEO/archive/shipsnet.json') as data_file:
    dataset = json.load(data_file)

shipsnet = pd.DataFrame(dataset)
x = np.array(dataset['data']).astype('uint8')
y = np.array(dataset['labels']).astype('uint8')
# Reshape input data to have shape (number_of_samples, height, width, channels)
x_reshaped = x.reshape(-1, 3, 80, 80).transpose([0, 2, 3, 1])  # Transpose to move the channel dimension to the last axis
y_reshaped = to_categorical(y, num_classes=2)

# Split data into train, validation, and test sets
x_train, x_test, y_train, y_test = train_test_split(x_reshaped, y_reshaped, test_size=0.20, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)


# Define UNet model
inp = Input((IMG_HEIGHT, IMG_WIDTH, BANDS))
e1,p1 = enc(inp, 16, 0.2)
e2, p2 = enc(p1, 32, 0.2)
e3, p3 = enc(p2, 64, 0.2)
e4, p4 = enc(p3, 128, 0.2, kernel_initializer=None)
e5, p5 = enc(p4, 256, 0.1)
b1 = conv_bl(p5, 512, 0.1)
d1 = dec(b1,e5, 256, 0.1)
d2 = dec(d1, e4, 128, 0.1, trainable=True)
d3 = dec(d2, e3, 64, 0.0, trainable=True)
d4 = dec(d3, e2, 32, 0.0, trainable=True)
d5 = dec(d4, e1, 16, 0.0, trainable=True)
o1 = conv_bl(d5, 8, 0.0, trainable=True)
o2 = conv_bl(o1,2, 0.0, trainable=True)
o3 = Conv2D(1, (3,3), (1,1), activation="sigmoid", padding="same")(o2)

# Compile the model
EPOCHS = 500
lr = 1e-4
batch_size = 16
unet = Model(inputs=[inp], outputs=[o3], name="Unet_for_ship")
unet.compile(optimizer=Adam(learning_rate=lr),
              loss="binary_crossentropy",
              metrics=["accuracy", iou_coef])


# Define callbacks
# mc = ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1)
# es = EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1, restore_best_weights=True)
# lrr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, min_delta=1e-4)

# Train the model
history = unet.fit(x_train, y_train, 
                   validation_data=(x_val, y_val), 
                   epochs=EPOCHS, 
                   batch_size=batch_size,
                   #callbacks=[mc, es, lrr]
                   )

# Evaluate the model
loss, accuracy = unet.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Plot training history
# Visualize predictions
# ...
