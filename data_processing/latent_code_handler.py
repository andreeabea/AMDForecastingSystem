import tensorflow as tf

from PIL import Image, ImageFilter

import numpy as np
from matplotlib import cm

from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K, preprocessing
from tensorflow.python.keras import Input


class LatentCodeHandler:

    def __init__(self, model_path):
        self.image_width = 256
        self.image_height = 256

        self.model = tf.keras.models.load_model(model_path)

        self.model.summary()

    def get_latent_codes(self, images):
        encoder_output = self.model.get_layer("encoded").output
        encoder_model = Model(self.model.input, encoder_output)
        latent_codes = encoder_model.predict(images)
        return latent_codes

    # TODO: not working - to predict next image in sequence
    def decode(self, latent_codes):
        encoder_output = self.model.get_layer("encoded").output
        encoder_model = Model(self.model.input, encoder_output)

        decoder_input = Input(shape=encoder_model.output.shape[1:])
        decoder_output = decoder_input
        norm_layer_count = 10
        for i in range(16, len(self.model.layers), 1):
            if i==21 or i==25 or i==29 or i==33 or i==37 or i==41:
                decoder_output = self.model.layers[i]([decoder_output, self.model.layers[i-norm_layer_count].output])
                norm_layer_count += 6
            else:
                decoder_output = self.model.layers[i](decoder_output)

        decoder = Model(decoder_input, decoder_output)

        prediction = decoder.predict(latent_codes)

        for i in range(len(prediction)):
            predictedImage = Image.fromarray((prediction[i].reshape(self.image_width, self.image_height)*255).astype(np.uint8))
            predictedImage.show("prediction")
            input()

    def get_test_images(self):
        testX = []
        # TODO: open file dialog to choose image
        image = Image.open("../data/Pacient 3/Vizita 2 - 05.12.2019/OD/602BBD40.tif").convert("L")
        #image = Image.open("../data/Pacient 1/Vizita 4 - 23.05.2018/OD/1764F070.tif").convert("L")
        #image = Image.open("../data/Pacient 25/Vizita 3 - 21.10.2019/OS/EEAD9E70.tif").convert("L")
        #image = Image.open("../data/Pacient 93/Vizita 3 - 27.03.2019/OD/36ADEF30.tif").convert("L")

        # resize img
        image = np.array(image.resize((self.image_width, self.image_height), Image.ANTIALIAS))
        image = image.reshape(self.image_width, self.image_height, 1)

        testX.append(image)

        testX = np.vstack(testX)
        testX = testX.reshape(-1, self.image_width, self.image_height, 1)
        # trainX, validX, testX = split_data()

        testX = testX / 255.0
        return testX

    def get_image(self, path):
        testX = []
        image = Image.open(path).convert("L")

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

    # code source: https://keras.io/examples/vision/grad_cam/
    def get_gradcam_heatmap(self, images):
        # we create a model that maps the input image to conv layer activations
        #5,7,8
        last_conv_layer = self.model.get_layer("conv2d_7")
        last_conv_layer_model = Model(self.model.inputs, last_conv_layer.output)

        # we create a model that maps the activations of the conv layer
        # to the latent embedding prediction
        classifier_input = Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        normalized = self.model.layers[12](x)
        flattened = self.model.layers[13](normalized)
        code = self.model.layers[14](flattened)
        classifier_model = Model(classifier_input, code)

        # compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            # Compute activations of the conv layer
            last_conv_layer_output = last_conv_layer_model(images)
            tape.watch(last_conv_layer_output)
            # Compute class predictions
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # gradient intensities
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # multiply each channel in the feature map array
        # by "how important this channel is"
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(last_conv_layer_output, axis=-1)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

        # Display heatmap
        plt.matshow(heatmap)
        plt.show()

        # We rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # We use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # We create an image with RGB colorized heatmap
        jet_heatmap = preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((images[0].shape[1], images[0].shape[0]))
        jet_heatmap = preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * 0.01 + images[0]
        superimposed_img = preprocessing.image.array_to_img(superimposed_img)

        # Display Grad CAM
        #superimposed_img.show("gradcam heatmap")
        superimposed_img.save('./heatmap.png')


if __name__ == '__main__':
    code_handler = LatentCodeHandler("../models/autoencoder256-original-sgm.h5")
    imgs = code_handler.get_test_images()
    #code_handler.reconstruct_images(imgs)
    code_handler.get_gradcam_heatmap(imgs)
