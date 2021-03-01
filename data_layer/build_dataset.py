# import the necessary packages
import os
from xml.dom import minidom

from PIL import Image, ImageOps

import config
import numpy as np

from code_handler import CodeHandler
from datetime import datetime


def create_sequences(eye_images, nb_features=257):
    # imagesX[i] will contain current image, imagesY[i] the next image in the sequence after imagesX[i]
    imagesX = []
    imagesY = []
    for i in range(len(eye_images) - 1):
        imagesX.append(eye_images[i])
        imagesY.append(eye_images[i+1])

    imagesX = np.array(imagesX)
    imagesY = np.array(imagesY)
    imagesX = imagesX.reshape(-1, nb_features)
    imagesY = imagesY.reshape(-1, nb_features)

    return imagesX, imagesY


def read_images_lstm(eyeData=None):
    # compute input data for LSTM model
    imagesX = []
    imagesY = []
    first = True
    x = -1
    y = -1

    code_handler = CodeHandler()

    for patient in os.scandir(config.ORIG_INPUT_DATASET):
        visits = len(list(os.scandir(patient)))
        # do not consider sequences with just 1 visit
        if (visits <= 1):
            continue

        left_eye_images = []
        right_eye_images = []

        left_eye_va = []
        right_eye_va = []

        for visit in os.scandir(patient):
            date = datetime.strptime(visit.name.split("- ")[1], '%d.%m.%Y')
            for eye in os.scandir(visit):
                # the scan of the retina fundus of the eye will not be added
                fundusImgPath = ""
                for file in os.scandir(eye):
                    if file.path.endswith(".xml"):
                        xmlfile = minidom.parse(file.path)
                        patientID = xmlfile.getElementsByTagName('ID')[0].firstChild.nodeValue
                        fundusImgPath = xmlfile.getElementsByTagName('ExamURL')[0].firstChild.nodeValue.rsplit("\\", 1)[1]
                for imageFile in os.scandir(eye):
                    # initially: fundusImgPath not in imageFile.name
                    if imageFile.path.endswith(".tif") and fundusImgPath not in imageFile.name:
                        image = Image.open(imageFile.path).convert("L")
                        # resize img
                        image = np.array(image.resize((256, 256), Image.ANTIALIAS))
                        # for the first version without resizing the images
                        if first:
                            x = image.shape[0]
                            y = image.shape[1]
                            first = False
                        if image.shape[0] == x and image.shape[1] == y:
                            image = image.reshape(x, y, 1)
                            if "OS" in eye.name:
                                if eyeData is not None:
                                    visual_acuity = find_visual_acuity(eyeData, patientID, "OS", date)
                                    if visual_acuity >= 0:
                                        left_eye_va.append(visual_acuity)
                                        left_eye_images.append(image)
                                else:
                                    left_eye_images.append(image)
                            else:
                                if eyeData is not None:
                                    visual_acuity = find_visual_acuity(eyeData, patientID, "OD", date)
                                    if visual_acuity >= 0:
                                        right_eye_va.append(visual_acuity)
                                        right_eye_images.append(image)
                                else:
                                    right_eye_images.append(image)
                        break

        if len(left_eye_images) > 1 and len(right_eye_images) > 1:
            left_eye_images = reshape_images(left_eye_images, x, y)
            left_eye_images = left_eye_images / 255
            right_eye_images = reshape_images(right_eye_images, x, y)
            right_eye_images = right_eye_images / 255

            left_eye_code_sequences = code_handler.get_latent_codes(left_eye_images)
            right_eye_code_sequences = code_handler.get_latent_codes(right_eye_images)

            if eyeData is not None:
                left_eye_code_sequences = append_va_to_features(left_eye_code_sequences, left_eye_va)
                right_eye_code_sequences = append_va_to_features(right_eye_code_sequences, right_eye_va)

            left_eye_imagesX, left_eye_imagesY = create_sequences(left_eye_code_sequences.tolist())
            right_eye_imagesX, right_eye_imagesY = create_sequences(right_eye_code_sequences.tolist())
            imagesX.append(left_eye_imagesX)
            imagesX.append(right_eye_imagesX)
            imagesY.append(left_eye_imagesY)
            imagesY.append(right_eye_imagesY)

    return x, y, imagesX, imagesY


def find_visual_acuity(eyeData, patientID, eye, date):
    id = patientID + eye
    for df in eyeData:
        if df['New ID'].iloc[0] == id:
            for i in range(len(df.index)):
                if df['Date'].iloc[i] == date:
                    return df['Eye'].iloc[i]
    return -1


def append_va_to_features(features, va_list):
    new_features = []
    for i in range(len(features)):
        new = np.append(features[i], va_list[i])
        new_features.append(new)

    return np.array(new_features)

# get images to feed autoencoder
def read_all_images():
    images = []
    x = 256
    y = 256

    for patient in os.scandir(config.ORIG_INPUT_DATASET):
        for visit in os.scandir(patient):
            for eye in os.scandir(visit):
                fundusImgPath = ""
                for file in os.scandir(eye):
                    if file.path.endswith(".xml"):
                        xmlfile = minidom.parse(file.path)
                        fundusImgPath = xmlfile.getElementsByTagName('ExamURL')[0].firstChild.nodeValue.rsplit("\\", 1)[1]
                for imageFile in os.scandir(eye):
                    if imageFile.path.endswith(".tif") and fundusImgPath not in imageFile.name:
                        with Image.open(imageFile.path) as image:
                            try:
                                image = ImageOps.grayscale(image)
                                #width, height = image.size
                                # image.show("initial")
                                #image = crop_retina_image(image, width, height)
                                # image.show("cropped")
                                # resize img
                                image = np.array(image.resize((x, y), Image.ANTIALIAS))
                                image = image.reshape(x, y, 1)
                                images.append(image)
                            except Exception:
                                print("Cannot convert" + imageFile.name)

    return x, y, images


def reshape_images(images, x, y):
    images = np.vstack(images)
    images = images.reshape(-1, x, y, 1)
    return images


def split_data_lstm(scaler=None, eyeData=None, dataX=None, dataY=None):
    if dataX is None and dataY is None and eyeData is None:
        # data is not given and VA should not be included
        x, y, dataX, dataY = read_images_lstm()
    else:
        if dataX is None and dataY is None and eyeData is not None:
            # include VA data from .csv
            x, y, dataX, dataY = read_images_lstm(eyeData)

    # otherwise (if dataX and dataY are not None) data is passed as parameters

    dataX = np.vstack(dataX)
    dataY = np.vstack(dataY)

    # if data needs to be scaled/normalized
    if scaler is not None:
        dataX = scaler.fit_transform(dataX)
        dataY = scaler.fit_transform(dataY)

    # compute the training split
    i = int(len(dataX) * config.TRAIN_SPLIT)
    trainX = dataX[:i]
    trainY = dataY[:i]

    # obtain the validation and testing splits
    valid = int(len(dataX) * config.VAL_SPLIT)
    validX = dataX[i:i+valid]
    validY = dataY[i:i+valid]

    testX = dataX[i+valid:]
    testY = dataY[i+valid:]

    return trainX, trainY, validX, validY, testX, testY


# split data for the autoencoder
def split_data():
    x, y, images = read_all_images()
    images = reshape_images(images, x, y)

    # compute the training split
    i = int(len(images) * config.TRAIN_SPLIT)
    train = images[:i]

    # obtain the validation and testing splits
    validLen = int(len(images) * config.VAL_SPLIT)
    valid = images[i:i+validLen]

    test = images[i+validLen:]

    return train, valid, test
