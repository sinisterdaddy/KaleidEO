# %%
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L

# %%
def conv_bl (inp, filters, dropout, activation="relu", kernel_initializer="he_normal", trainable = False):
    con1 = Conv2D(filters, (3,3), activation= activation, kernel_initializer=kernel_initializer, padding="same",)(inp)
    d1 = Dropout(dropout)(con1)
    con1 = Conv2D(filters, (3,3), activation= activation, kernel_initializer=kernel_initializer,padding="same", trainable = trainable)(d1)
    return con1

def enc(inp, filters, dropout, activation="relu", kernel_initializer="he_normal", trainable = False):
    c1 = conv_bl(inp, filters, dropout, activation = activation, kernel_initializer= kernel_initializer)
    c2 = conv_bl(c1, filters, dropout, activation = activation, kernel_initializer= kernel_initializer, trainable = trainable)
    p1 = MaxPooling2D((2,2))(c2)
    return c2, p1

def dec(inp, skip, filters, dropout, activation="relu", kernel_initializer="he_normal", trainable = False):
    up1 = Conv2DTranspose(filters, (2,2), (2,2))(inp)
    s1 = concatenate([up1, skip])
    c1 = conv_bl(s1, filters, dropout, activation = activation, kernel_initializer= kernel_initializer, trainable = trainable)
    return c1

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

# %%
## Define constants
IMG_HEIGHT = 256
IMG_WIDTH = 256
BANDS = 3
inp_path = "" 
model_path = ""

# %%
### Define architecture

## Input
inp = Input((IMG_HEIGHT, IMG_WIDTH, BANDS))

## encoder_block          ##Output after pooling
e1,p1 = enc(inp, 16, 0.2) ##128,128,16
e2, p2 = enc(p1, 32, 0.2)  ##64,64,32
e3, p3 = enc(p2, 64, 0.2)  ##32,32,64
e4, p4 = enc(p3, 128, 0.2, kernel_initializer=None) ##16,16,128
e5, p5 = enc(p4, 256, 0.1) ##8,8,256

## base block
b1 = conv_bl(p5, 512, 0.1) ##8,8,512

## decoder block
d1 = dec(b1,e5, 256, 0.1, )
d2 = dec(d1, e4, 128, 0.1, trainable = True)
d3 = dec(d2, e3, 64, 0.0,trainable = True)
d4 = dec(d3, e2, 32, 0.0, trainable = True)
d5 = dec(d4, e1, 16, 0.0, trainable= True)

## output block
o1 = conv_bl(d5, 8, 0.0, trainable  = True)
o2 = conv_bl(o1,2, 0.0, trainable = True)
o3 = Conv2D(1, (3,3), (1,1), activation = "sigmoid", padding= "same" )(o2)


# %%
### Define hyperparameters
EPOCHS = 500
lr=1e-4
batch_size = 16

# %%
### Define callbacks
# mc = ModelCheckpoint(model_path, monitor = "val_loss", save_best_only =True, mode = "min", verbose = 1 )
# es = EarlyStopping(monitor = "val_loss", min_delta = 0, patience = 5, verbose = 1, restore_best_weights = True)
# lrr = ReduceLROnPlateau(monitor= "val_loss", factor = 0.1, patience = 5, verbose = 1, min_delta = 1e-4)

# %%
unet = Model(inputs=[inp], outputs = [o3], name="Unet_for_ship")
unet.compile(optimizer = Adam(learning_rate=lr), loss="binary_crossentropy", metrics = ["accuracy"])


# %%
eff = EfficientNetV2L(input_shape=(256,256,3), weights = "imagenet", include_top = False)

# %%
lay1 = eff.layers[19]
lay1w = lay1.get_weights()
print(lay1w[0].shape)
lay2 = eff.layers[4]
print(lay2)
lay2w = lay2.get_weights()


# %%
lay3 = eff.layers[5]
lay3w = lay3.get_weights()

# %%
lay3w

# %%
# Assuming you want to use default bias weights (all zeros)
kernel_weights = lay1w[0]  # Extract kernel weights
input_shape = unet.input_shape[1:]  # Get input shape of the model's input tensor
output_channels = kernel_weights.shape[-1]  # Get the number of output channels

# Initialize bias weights with zeros
bias_weights = np.zeros((output_channels,))
ulay1 = unet.get_layer(index = 13)

# Set weights for ulay1
ulay1.set_weights([kernel_weights, bias_weights])

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3]) + K.sum(y_pred,[1,2,3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

# Compile the model with the IoU metric
unet.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy", iou_coef])


# %%
# # test_con =  Conv2D(32, (3,3), (1,1), activation = "relu", padding="same", weights =  [lay1w])(p3)
# test_con =  Conv2D(32, (3,3), (1,1), activation = "relu", padding="same", weights =  [lay1w])(p3)
# test_con.set_weights(lay1w)

# %%
#unet.summary()

# %%
#eff.summary()

# %%
### Define and compile model

# unet = Model(inputs=[inp], outputs = [o3], name="Unet_for_ship")
# unet.compile(optimizer = Adam(learning_rate=lr), loss="binary_crossentropy", metrics = ["accuracy"])

history = unet.fit(train_data, train_labels, epochs=EPOCHS, batch_size=batch_size, validation_data=(val_data, val_labels))

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
# for layers in unet.layers:
#     print(layers)

# %%
