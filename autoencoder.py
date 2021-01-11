import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Convolution2D, BatchNormalization, UpSampling2D, Concatenate, Flatten, Dense, Reshape

import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
import numpy as np

from build_dataset import split_data

trainX, validX, testX = split_data()

trainX = trainX / 255.0
validX = validX / 255.0
testX = testX / 255.0

print(len(trainX))

# initially tried with 16
latentDim = 256

# define model
input_layer = Input(trainX[0].shape)

# Encoder
down_1 = Convolution2D(64, kernel_size=4, strides=2, padding='same', activation="relu")(input_layer)

down_2 = Convolution2D(64 * 2, kernel_size=4, strides=2, padding='same', activation="relu")(down_1)
norm_2 = BatchNormalization()(down_2)

down_3 = Convolution2D(64 * 4, kernel_size=4, strides=2, padding='same', activation="relu")(norm_2)
norm_3 = BatchNormalization()(down_3)

down_4 = Convolution2D(64 * 8, kernel_size=4, strides=2, padding='same', activation="relu")(norm_3)
norm_4 = BatchNormalization()(down_4)

down_5 = Convolution2D(64 * 8, kernel_size=4, strides=2, padding='same', activation="relu")(norm_4)
norm_5 = BatchNormalization()(down_5)

down_6 = Convolution2D(64 * 8, kernel_size=4, strides=2, padding='same', activation="relu")(norm_5)
norm_6 = BatchNormalization()(down_6)

down_7 = Convolution2D(64 * 8, kernel_size=4, strides=2, padding='same', activation="relu")(norm_6)
norm_7 = BatchNormalization()(down_7)

volumeSize = backend.int_shape(norm_7)
flattened = Flatten()(norm_7)
latent = Dense(latentDim, name="encoded", activation="relu")(flattened)

dense = Dense(np.prod(volumeSize[1:]), activation="relu")(latent)
reshaped = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(dense)

# Decoder
upsample_1 = UpSampling2D(size=2)(reshaped)
up_conv_1 = Convolution2D(64 * 8, kernel_size=4, strides=1, padding='same', activation='relu')(upsample_1)
norm_up_1 = BatchNormalization(momentum=0.8)(up_conv_1)
add_skip_1 = Concatenate()([norm_up_1, norm_6])

upsample_2 = UpSampling2D(size=2)(add_skip_1)
up_conv_2 = Convolution2D(64 * 8, kernel_size=4, strides=1, padding='same', activation='relu')(upsample_2)
norm_up_2 = BatchNormalization(momentum=0.8)(up_conv_2)
add_skip_2 = Concatenate()([norm_up_2, norm_5])

upsample_3 = UpSampling2D(size=2)(add_skip_2)
up_conv_3 = Convolution2D(64 * 8, kernel_size=4, strides=1, padding='same', activation='relu')(upsample_3)
norm_up_3 = BatchNormalization(momentum=0.8)(up_conv_3)
add_skip_3 = Concatenate()([norm_up_3, norm_4])

upsample_4 = UpSampling2D(size=2)(add_skip_3)
up_conv_4 = Convolution2D(64 * 4, kernel_size=4, strides=1, padding='same', activation='relu')(upsample_4)
norm_up_4 = BatchNormalization(momentum=0.8)(up_conv_4)
add_skip_4 = Concatenate()([norm_up_4, norm_3])

upsample_5 = UpSampling2D(size=2)(add_skip_4)
up_conv_5 = Convolution2D(64 * 2, kernel_size=4, strides=1, padding='same', activation='relu')(upsample_5)
norm_up_5 = BatchNormalization(momentum=0.8)(up_conv_5)
add_skip_5 = Concatenate()([norm_up_5, norm_2])

upsample_6 = UpSampling2D(size=2)(add_skip_5)
up_conv_6 = Convolution2D(64 * 2, kernel_size=4, strides=1, padding='same', activation='relu')(upsample_6)
norm_up_6 = BatchNormalization(momentum=0.8)(up_conv_6)
add_skip_6 = Concatenate()([norm_up_6, down_1])

# last upsample and output layer
last_upsample = UpSampling2D(size=2)(add_skip_6)
output_layer = Convolution2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(last_upsample)

model = Model(input_layer, output_layer)

# define callbacks
my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='models/autoencoder256-original-sgm.h5'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
]

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001, beta_1=0.5, decay=1e-5), metrics=['mean_squared_error'])

# fit model
model.fit(trainX, trainX, epochs=100, batch_size=32, verbose=1, callbacks=my_callbacks, validation_data=(validX, validX))

plt.plot(model.history.history['loss'], label = 'Train loss', alpha = .5)
plt.plot(model.history.history['val_loss'], label = 'Val. loss', alpha = .5)
plt.title('Linear model loss')
plt.legend()
plt.show()

plt.savefig('./plots/autoenc-loss-256-original-sgm.png')

# evaluate model
print("Evaluate on test data")
results = model.evaluate(testX, testX)
print("test loss:", results[0])

# view model architecture
model.summary()
