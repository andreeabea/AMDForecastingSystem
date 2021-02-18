import tensorflow as tf

from PIL import Image, ImageFilter

import numpy as np

from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
#import innvestigate


class CodeHandler:

    def __init__(self):
        self.image_width = 256
        self.image_height = 256

        self.model = tf.keras.models.load_model('models/autoencoder256-original-sgm.h5')

    def get_latent_codes(self, images):
        encoder_output = self.model.get_layer("encoded").output

        encoder_model = Model(self.model.input, encoder_output)
        latent_codes = encoder_model.predict(images)

        return latent_codes

    def get_test_images(self):
        testX = []
        image = Image.open("./data/Pacient 3/Vizita 2 - 05.12.2019/OD/604D00E0.tif").convert("L")

        # resize img
        image = np.array(image.resize((self.image_width, self.image_height), Image.ANTIALIAS))
        image = image.reshape(self.image_width, self.image_height, 1)

        testX.append(image)

        testX = np.vstack(testX)
        testX = testX.reshape(-1, self.image_width, self.image_height, 1)
        # trainX, validX, testX = split_data()

        testX = testX / 255.0
        return testX

    def reconstruct_images(self, testX):
        prediction = self.model.predict(testX)

        for i in range(len(prediction)):
            predictedImage = Image.fromarray((prediction[i].reshape(self.image_width, self.image_height)*255).astype(np.uint8))
            Image.fromarray((testX[i].reshape(self.image_width, self.image_height)*255).astype(np.uint8)).show("actual")
            predictedImage.show("prediction")
            input()

    # def investigate_autoencoder(images):
    #     analyzer = innvestigate.create_analyzer("deep_taylor", model)
    #     x=images[0]
    #     x = x[None, :, :, :]
    #     analysis = analyzer.analyze(x)
    #
    #     analysis = np.array([0.6, 0.4])
    #
    #     # Aggregate along color channels and normalize to [-1, 1]
    #     analysis = analysis.sum(axis=np.argmax(np.asarray(analysis.shape) == 3))
    #     analysis /= np.max(np.abs(analysis))
    #     # Plot
    #     plt.imshow(analysis[0], cmap="seismic", clim=(-1, 1))


if __name__ == '__main__':
    code_handler = CodeHandler()

    test = code_handler.get_test_images()
    code_handler.reconstruct_images(test)

    #investigate_autoencoder(test)
