import cv2
import numpy as np
from keras import models
import tensorflow as tf
from tensorflow import keras
import sys
from PIL import Image
import face_recognition
import matplotlib.pyplot as plt

image_path = "/Users/christianbv/Desktop/66666577_10157282576823397_5693975617620410368_n.jpg"
savePath = "/Users/christianbv/PycharmProjects/ML/venv/conv_layer_visualizations"

def load_model():
    model_path = "/Users/christianbv/PycharmProjects/ML/venv/models_model_adam_v2_14_11"
    model = models.load_model(model_path)
    print(model.summary())
    return model

def convert_model(model):
    # redefine model to output right after the first hidden layer
    ixs = [1, 4, 7, 10, 13]
    outputs = [model.layers[i + 1].output for i in ixs]
    converted_model = keras.Model(inputs=model.inputs, outputs=outputs)
    print("Printer converted model:")
    print(converted_model.summary())
    print(converted_model.layers)
    return converted_model

def convert_model2(model):
    model = keras.Model(inputs=model.inputs, outputs=model.layers[4].output)
    print("Printer converted 2")
    print(model.summary())
    return model

"""
    Finds the faces in each image, and crops the images around that face
    Assumes the face detection software finds exactly one face in each image
"""
def find_face_and_crop(image, name):
    face_locations = face_recognition.face_locations(image)
    im = Image.fromarray(image)
    faces_in_image = []
    for (top,right,bottom,left) in face_locations:
        padding = abs(left - right) * 0.4
        im_cropped = im.crop((left - padding, top - padding, right + padding, bottom + padding))
        im_resized = im_cropped.resize((256, 256), Image.ANTIALIAS)
        faces_in_image.append(im_resized)
    if len(faces_in_image) != 1:
        return None

    savefig = faces_in_image[0]
    savefig.save(f'{savePath}/{name}/cropped_img.png')
    return faces_in_image[0]

def load_image(path):
    image = Image.open(path)
    img_arr = np.asarray(image)
    return img_arr

def obtain_feature_maps(model, img):
    im_array = keras.preprocessing.image.img_to_array(img)
    im_array = tf.expand_dims(im_array, 0)  # Create a batch

    feature_maps = model.predict(im_array)
    return feature_maps

def plot_feature_maps(feature_maps, name):
    # plot the output from each block
    square = 4
    for i, fmap in enumerate(feature_maps):
        # plot all 64 maps in an 8x8 squares
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                #ax = plt.subplot(square, square, ix)
                ax = plt.subplot(4,4,ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                #plt.imshow(fmap[0, :, :, ix - 1], cmap='gray')
                plt.imshow(fmap[0, :, :, ix - 1])

                ix += 1
        # Save figure
        folder = "/Users/christianbv/PycharmProjects/ML/venv/conv_layer_visualizations"
        plt.savefig(f'{folder}/{name}/layer_{i}')

        # show the figure
        plt.show()


def run():
    model = load_model()
    converted_model = convert_model(model)
    image_paths = ["/Users/christianbv/Desktop/78543013_3221426351217323_7771579877131550720_o.jpg",
                   "/Users/christianbv/Desktop/66666577_10157282576823397_5693975617620410368_n.jpg",
                   "/Users/christianbv/Desktop/IMG_0470.JPG"]
    names = ["Adrian", "Christian", "Jacob"]
    for img, name in zip(image_paths, names):
        img = load_image(img)
        cropped_img = find_face_and_crop(img, name)
        feature_maps = obtain_feature_maps(converted_model, cropped_img)
        plot_feature_maps(feature_maps, name)


run()

