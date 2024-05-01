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
from keras.utils import to_categorical
import numpy as np
from numpy import expand_dims
import pandas as pd
import json
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import keras
from keras import layers
from scikeras.wrappers import KerasClassifier

#from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.src.legacy.preprocessing.image import ImageDataGenerator


#from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
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
import cv2

def resize_images(images):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, (256, 256))
        resized_images.append(resized_img)
    return np.array(resized_images)
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
unet = Model(inputs=[inp], outputs = [o3], name="Unet_for_ship")
unet.compile(optimizer=Adam(learning_rate=lr),
              loss="binary_crossentropy",
              metrics=["accuracy", iou_coef])
input_shape = unet.input_shape[1:]  # Get input shape of the model's input tensor
output_channels = kernel_weights.shape[-1]  # Get the number of output channels

# Initialize bias weights with zeros
bias_weights = np.zeros((output_channels,))
ulay1 = unet.get_layer(index = 13)

# Set weights for ulay1
ulay1.set_weights([kernel_weights, bias_weights])



with open('C:/Users/krish/OneDrive/Documents/GitHub/KaleidEO/archive/shipsnet.json') as data_file:
    dataset = json.load(data_file)
shipsnet= pd.DataFrame(dataset)
shipsnet.head(1000)
print(len(shipsnet['data'][0]))

# %%
shipsnet.info()

# %%
shipsnet = shipsnet[["data","labels"]]
shipsnet.head()

# %%
ship_images = shipsnet["labels"].value_counts()[0]
no_ship_images = shipsnet["labels"].value_counts()[1]
print("Number of the ship_images :{}".format(ship_images),"\n")
print("Number of the ship_images :{}".format(no_ship_images))

# %%
# Turning the json information into numpy array and then assign it as x and y variables
x = np.array(dataset['data']).astype('uint8')
y = np.array(dataset['labels']).astype('uint8')

# %%
x.shape

# %%
x

# %%
x_reshaped = x.reshape([-1, 3, 80, 80])

# %%
x_reshaped

# %%
x_reshaped[0].shape

# %%
print(x_reshaped[0][0])

# %%
x_reshaped[0][0].shape

# %%
print(x_reshaped[0][1])

# %%
print(x_reshaped[0])

# %%
x_reshaped[0][0].shape

# %%
x_reshaped = x.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])
x_reshaped.shape

# %%
print(x_reshaped[0][79])

# %%
x_reshaped[0][4].shape

# %%
y.shape

# %%
y_reshaped = to_categorical(y, num_classes=2)

y_reshaped.shape

# %%
y_reshaped

# %%
"""
## Exploring the images
"""

# %%
image_no_ship = x_reshaped[y==0]
image_ship = x_reshaped[y==1]

def plot(a,b):
    
    plt.figure(figsize=(15, 15))
    for i, k in enumerate(range(1,9)):
        if i < 4:
            plt.subplot(2,4,k)
            plt.title('Not A Ship')
            plt.imshow(image_no_ship[i+2])
            plt.axis("off")
        else:
            plt.subplot(2,4,k)
            plt.title('Ship')
            plt.imshow(image_ship[i+15])
            plt.axis("off")
            
    plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0.25)

#Implementation of the function 

plot(image_no_ship, image_ship)

# %%
# def plotHistogram(ship, not_ship):

#     plt.figure(figsize = (10,7))
#     plt.subplot(2,2,1)
#     plt.imshow(ship)
#     plt.axis('off')
#     plt.title('Ship')
#     histo = plt.subplot(2,2,2)
#     histo.set_ylabel('Count', fontweight = "bold")
#     histo.set_xlabel('Pixel Intensity', fontweight = "bold")
#     n_bins = 30
#     plt.hist(ship[:,:,0].flatten(), bins = n_bins, lw = 0, color = 'r', alpha = 0.5);
#     plt.hist(ship[:,:,1].flatten(), bins = n_bins, lw = 0, color = 'g', alpha = 0.5);
#     plt.hist(ship[:,:,2].flatten(), bins = n_bins, lw = 0, color = 'b', alpha = 0.5);
#     plt.show()
#     print("Minimum pixel value of this image: {}".format(ship.min()))
#     print("Maximum pixel value of this image: {}".format(ship.max()))
#     plt.figure(figsize = (10,7))
#     plt.subplot(2,2,3)
#     plt.imshow(not_ship)
#     plt.axis('off')
#     plt.title('Not A Ship')
#     histo = plt.subplot(2,2,4)
#     histo.set_ylabel('Count', fontweight = "bold")
#     histo.set_xlabel('Pixel Intensity', fontweight = "bold")
#     n_bins = 30
#     plt.hist(not_ship[:,:,0].flatten(), bins = n_bins, lw = 0, color = 'r', alpha = 0.5);
#     plt.hist(not_ship[:,:,1].flatten(), bins = n_bins, lw = 0, color = 'g', alpha = 0.5);
#     plt.hist(not_ship[:,:,2].flatten(), bins = n_bins, lw = 0, color = 'b', alpha = 0.5);
#     plt.show()
#     print("Minimum pixel value of this image: {}".format(not_ship.min()))
#     print("Maximum pixel value of this image: {}".format(not_ship.max()))
# #Implementation of the function

# for i in range (10,14):
#     plotHistogram(x_reshaped[y==1][i], x_reshaped[y==0][i])

# # %%
# my_list = [(0, 'R channel'), (1, 'G channel'), (2, 'B channel')]

# plt.figure(figsize = (15,15))

# for i, k in my_list:
#     plt.subplot(1,3,i+1)
#     plt.title(k)
#     plt.ylabel('Height {}'.format(x_reshaped[y==0][5].shape[0]))
#     plt.xlabel('Width {}'.format(x_reshaped[y==0][5].shape[1]))
#     plt.imshow(x_reshaped[y==0][5][ : , : , i])

# %%
# my_list = [(0, 'R channel'), (1, 'G channel'), (2, 'B channel')]

# plt.figure(figsize = (15,15))

# for i, k in my_list:
#     plt.subplot(1,3,i+1)
#     plt.title(k)
#     plt.ylabel('Height {}'.format(x_reshaped[y==0][5].shape[0]))
#     plt.xlabel('Width {}'.format(x_reshaped[y==0][5].shape[1]))
#     plt.imshow(x_reshaped[y==1][5][ : , : , i])

# %%
"""
## Modelling
"""

# %%
x_reshaped = x_reshaped / 255

# %%
x_reshaped[0][0][0] # Normalized RGB values of the firs pixel of the first image in the dataset.

# %%
n_bins = 30
# plt.hist(x_reshaped[y == 0][0][:,:,0].flatten(), bins = n_bins, lw = 0, color = 'r', alpha = 0.5);
# plt.hist(x_reshaped[y == 0][0][:,:,1].flatten(), bins = n_bins, lw = 0, color = 'g', alpha = 0.5);
# plt.hist(x_reshaped[y == 0][0][:,:,2].flatten(), bins = n_bins, lw = 0, color = 'b', alpha = 0.5);
# plt.ylabel('Count', fontweight = "bold")
# plt.xlabel('Pixel Intensity', fontweight = "bold")
# plt.title("Histogram of normalized data")
# plt.show()
x_train_1, x_test, y_train_1, y_test = train_test_split(x_reshaped, y_reshaped,
                                                        test_size = 0.20, random_state = 42)


x_train, x_val, y_train, y_val = train_test_split(x_train_1, y_train_1, 
                                                  test_size = 0.25, random_state = 42)


print("x_train shape",x_train.shape)
print("x_test shape",x_test.shape)
print("y_train shape",y_train.shape)
print("y_test shape",y_test.shape)
print("y_train shape",x_val.shape)
print("y_test shape",y_val.shape)
print("x_train_1 shape",x_train_1.shape)
print("x_val shape",x_val.shape)
print("y_train_1 shape",y_train_1.shape)


# %%
x_test.shape

# %%
x_train.shape

# %%
# # test_con =  Conv2D(32, (3,3), (1,1), activation = "relu", padding="same", weights =  [lay1w])(p3)
# test_con =  Conv2D(32, (3,3), (1,1), activation = "relu", padding="same", weights =  [lay1w])(p3)
# test_con.set_weights(lay1w)

# %%
### Define and compile model
history = unet.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=batch_size)
# %%
#unet.summary()

# %%
#eff.summary()



# unet = Model(inputs=[inp], outputs = [o3], name="Unet_for_ship")
# unet.compile(optimizer = Adam(learning_rate=lr), loss="binary_crossentropy", metrics = ["accuracy"])


# # %%
# for layers in unet.layers:
#     print(layers)


# Evaluate the model
loss, accuracy = unet.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)# %%








# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# import numpy as np
# import matplotlib.pyplot as plt
# from keras import backend as K
# from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
# from keras.utils import to_categorical
# import pandas as pd
# import json

# from sklearn.model_selection import train_test_split

# # Load data and preprocess
# with open('C:/Users/krish/OneDrive/Documents/GitHub/KaleidEO/archive/shipsnet.json') as data_file:
#     dataset = json.load(data_file)

# shipsnet = pd.DataFrame(dataset)
# x = np.array(dataset['data']).astype('uint8')
# y = np.array(dataset['labels']).astype('uint8')

# x_reshaped = x.reshape([-1, 3, 80, 80]).transpose([0, 2, 3, 1])
# x_reshaped = x_reshaped / 255

# # Modify target labels to be binary masks
# y_reshaped = np.expand_dims(y, axis=-1)  # Add an extra dimension to match the model output shape

# # Split data into train, validation, and test sets
# x_train_1, x_test, y_train_1, y_test = train_test_split(x_reshaped, y_reshaped, test_size=0.20, random_state=42)
# x_train, x_val, y_train, y_val = train_test_split(x_train_1, y_train_1, test_size=0.25, random_state=42)


# # Resize images to match model input size
# from keras.preprocessing.image import img_to_array, array_to_img

# x_train_resized = []
# for img in x_train:
#     img_resized = array_to_img(img).resize((256, 256))
#     x_train_resized.append(img_to_array(img_resized))
# x_train_resized = np.array(x_train_resized)

# x_val_resized = []
# for img in x_val:
#     img_resized = array_to_img(img).resize((256, 256))
#     x_val_resized.append(img_to_array(img_resized))
# x_val_resized = np.array(x_val_resized)

# x_test_resized = []
# for img in x_test:
#     img_resized = array_to_img(img).resize((256, 256))
#     x_test_resized.append(img_to_array(img_resized))
# x_test_resized = np.array(x_test_resized)

# # Define U-Net architecture
# def conv_bl(inp, filters, dropout, activation="relu", kernel_initializer="he_normal", trainable=False):
#     con1 = Conv2D(filters, (3,3), activation=activation, kernel_initializer=kernel_initializer, padding="same")(inp)
#     d1 = Dropout(dropout)(con1)
#     con1 = Conv2D(filters, (3,3), activation=activation, kernel_initializer=kernel_initializer, padding="same", trainable=trainable)(d1)
#     return con1

# def enc(inp, filters, dropout, activation="relu", kernel_initializer="he_normal", trainable=False):
#     c1 = conv_bl(inp, filters, dropout, activation=activation, kernel_initializer=kernel_initializer)
#     c2 = conv_bl(c1, filters, dropout, activation=activation, kernel_initializer=kernel_initializer, trainable=trainable)
#     p1 = MaxPooling2D((2,2))(c2)
#     return c2, p1

# def dec(inp, skip, filters, dropout, activation="relu", kernel_initializer="he_normal", trainable=False):
#     up1 = Conv2DTranspose(filters, (2,2), (2,2))(inp)
#     s1 = concatenate([up1, skip])
#     c1 = conv_bl(s1, filters, dropout, activation=activation, kernel_initializer=kernel_initializer, trainable=trainable)
#     return c1

# def iou_coef(y_true, y_pred, smooth=1):
#     intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
#     union = K.sum(y_true,[1,2,3]) + K.sum(y_pred,[1,2,3]) - intersection
#     iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
#     return iou

# IMG_HEIGHT = 256
# IMG_WIDTH = 256
# BANDS = 3
# inp = Input((IMG_HEIGHT, IMG_WIDTH, BANDS))
# e1, p1 = enc(inp, 16, 0.2)
# e2, p2 = enc(p1, 32, 0.2)
# e3, p3 = enc(p2, 64, 0.2)
# e4, p4 = enc(p3, 128, 0.2, kernel_initializer=None)
# e5, p5 = enc(p4, 256, 0.1)
# b1 = conv_bl(p5, 512, 0.1)
# d1 = dec(b1, e5, 256, 0.1)
# d2 = dec(d1, e4, 128, 0.1, trainable=True)
# d3 = dec(d2, e3, 64, 0.0, trainable=True)
# d4 = dec(d3, e2, 32, 0.0, trainable=True)
# d5 = dec(d4, e1, 16, 0.0, trainable=True)
# o1 = conv_bl(d5, 8, 0.0, trainable=True)
# o2 = conv_bl(o1, 2, 0.0, trainable=True)
# o3 = Conv2D(1, (3,3), (1,1), activation="sigmoid", padding="same")(o2)

# # Define hyperparameters
# EPOCHS = 500
# lr = 1e-4
# batch_size = 16

# # Compile the model
# unet = Model(inputs=[inp], outputs=[o3], name="Unet_for_ship")
# unet.compile(optimizer=Adam(learning_rate=lr),
#               loss="binary_crossentropy",
#               metrics=["accuracy", iou_coef])

# # Train the model
# history = unet.fit(x_train_resized, y_train, validation_data=(x_val_resized, y_val), epochs=EPOCHS, batch_size=batch_size)

# # Evaluate the model
# loss, accuracy = unet.evaluate(x_test_resized, y_test)
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)
