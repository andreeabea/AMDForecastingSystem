import tensorflow as tf

from PIL import Image, ImageFilter

import numpy as np

from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
#import innvestigate


x = 256
y = 256

model = tf.keras.models.load_model('models/autoencoder256-original-sgm.h5')
#model.summary()


def get_latent_codes(images):
    encoder_output = model.get_layer("encoded").output

    encoder_model = Model(model.input, encoder_output)
    latent_codes = encoder_model.predict(images)
    return latent_codes


def get_test_images():
    testX = []
    image = Image.open("./data/Pacient 3/Vizita 2 - 05.12.2019/OD/604D00E0.tif").convert("L")
    # image = image.filter(ImageFilter.GaussianBlur)
    # image.show("actual")
    width, height = image.size
    # image = crop_retina_image(image, width, height)

    # resize img
    image = np.array(image.resize((x, y), Image.ANTIALIAS))
    image = image.reshape(x, y, 1)

    testX.append(image)

    testX = np.vstack(testX)
    testX = testX.reshape(-1, x, y, 1)
    # trainX, validX, testX = split_data()

    testX = testX / 255.0
    return testX


def image_reconstruction(testX):
    prediction = model.predict(testX)

    for i in range(len(prediction)):
        predictedImage = Image.fromarray((prediction[i].reshape(x, y)*255).astype(np.uint8))
        Image.fromarray((testX[i].reshape(x, y)*255).astype(np.uint8)).show("actual")
        predictedImage.show("prediction")
        input()


def investigate_autoencoder(images):
    analyzer = innvestigate.create_analyzer("deep_taylor", model)
    x=images[0]
    x = x[None, :, :, :]
    analysis = analyzer.analyze(x)

    analysis = np.array([0.6, 0.4])

    # Aggregate along color channels and normalize to [-1, 1]
    analysis = analysis.sum(axis=np.argmax(np.asarray(analysis.shape) == 3))
    analysis /= np.max(np.abs(analysis))
    # Plot
    plt.imshow(analysis[0], cmap="seismic", clim=(-1, 1))

#test = get_test_images()
#image_reconstruction(test)
#??
#investigate_autoencoder(test)

