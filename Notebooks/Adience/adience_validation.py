import numpy as np
from keras import models
import tensorflow as tf
from tensorflow import keras
import sys
import face_recognition
import pandas as pd
from PIL import Image
from os import walk
from os import listdir
import shutil
import os

project_path = "/Users/christianbv/PycharmProjects/ML/venv/"

# Loading the model for inference
model_path = "/Users/christianbv/PycharmProjects/ML/venv/models_model_adam_v2_14_11"
model = models.load_model(model_path)

# Datapath
faces_path = "/Users/christianbv/PycharmProjects/ML/venv/faces"
fold_folder_path = "/Users/christianbv/PycharmProjects/ML/venv/fold_faces" # Empty folder with only images for this given fold

def get_folds_available():
    files = listdir(project_path)
    fold_filenames = [x for x in files if ".txt" in x]
    fold_csv = pd.read_csv(fold_filenames[0], sep="\t")
    for foo in range(1,len(fold_filenames)):
        fold_csv = fold_csv.append(pd.read_csv(fold_filenames[foo], sep = "\t"))
    return fold_csv

# Finding the images for the given fold
fold_csv = get_folds_available()
fold_csv["folds"] = fold_csv["user_id"].apply(lambda x: x.split("@")[1]) # Obtaining the fold

# Our own model!
class_labels = ['0-2', '3-6', '7-12', '13-17', '18-22', '23-26', '27-33', '34-44', '45-59', '60-199']

# Adience class_labels:
adience_class_labels = ['0-2', '4-6', '8-13', '15-20', '25-32','38-43','48-53', '60-100']
out_of_interval_years = set({3,7,14,21,22,23,24,33,34,35,36,37,44,45,46,47,54,55,56,57,58,59})

# Obtaining the path to all images
def create_fold_faces_folder():
    paths = []
    all_filenames = []
    for (dirpath, dirnames, filenames) in walk (faces_path):
        for dirname in dirnames:
            fold_name = dirname.split("@")[1]
            for filename in listdir(faces_path+"/"+dirname+"/"):
                if ".jpg" in filename:
                    # Creating folder with only fold images
                    f = (".").join(filename.split(".")[2:])
                    if f in fold_csv["original_image"].values:
                        shutil.move(faces_path + "/"+ dirname + "/" + filename, fold_folder_path + "/" +f)


def predict_acc_age(predictions):
    score = 0
    avg_age = [1, 4.5, 9.5, 15, 20, 24.5, 30, 39, 52, 75]
    for i in range(len(predictions)):
        score += avg_age[i] * predictions[i]
    return int(score)

"""
    Predicts the age for a given picture
    Returns:
        predicted age (regression), predicted age class (class with highest softmax value), and prob (highest softmax value).
"""
def predict_age(im):
    im_array = keras.preprocessing.image.img_to_array(im)
    im_array = tf.expand_dims(im_array, 0)  # Create a batch

    predictions = model.predict(im_array)
    score = tf.nn.softmax(predictions[0])
    prob = max(score.numpy())
    predicted_age_class = class_labels[np.argmax(score.numpy())]
    predicted_age = predict_acc_age(score.numpy())
    return predicted_age, predicted_age_class, prob

"""
    Finds the faces in each image, and crops the images around that face
    Assumes the face detection software finds exactly one face in each image
"""
def find_face_and_crop(image):
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
    return faces_in_image[0]

def max_and_min_year(label):
    label = label[1:-1].split(",")
    try:
        min, max = int(label[0].strip(" ")), int(label[1].strip(" "))
    except ValueError:
        print(label)
        sys.exit()
    return min, max

def max_and_min_year_2(label):
    label = label.split("-")
    return int(label[0]), int(label[1])

def exact_accuracy(min,max,predicted_age):
    return min <= predicted_age <= max

def one_off_accuracy(predicted_age, label):
    # Obtain index of label from adience_class:
    min,max = max_and_min_year(label)
    label = f'{min}-{max}'
    before_bool, after_bool, curr_bool = False, False, False
    for index, interval in enumerate(adience_class_labels):
        if label == interval:
            before = adience_class_labels[index-1] if index != 0 else None
            after = adience_class_labels[index+1] if index != len(adience_class_labels)-1 else None
            if before:
                min_b, max_b = max_and_min_year_2(before)
                before_bool = exact_accuracy(min_b, max_b, predicted_age)
            if after:
                min_a, max_a = max_and_min_year_2(after)
                after_bool = exact_accuracy(min_a, max_a, predicted_age)
            min_curr, max_curr = max_and_min_year_2(adience_class_labels[index])
            curr_bool = exact_accuracy(min_curr, max_curr, predicted_age)
            break

    return before_bool or after_bool or curr_bool

def run():
    out_of_years = 0
    exact_hits = 0
    one_off_hits = 0
    n = 0

    for image in fold_csv.values:
        file = image[1]
        label = image[3]
        if "," not in label:
            continue
        min, max = max_and_min_year(label)
        path = fold_folder_path + "/" + file
        if not os.path.isfile(path):
            continue
        image = Image.open(path)
        img_arr = np.asarray(image)
        cropped_img = find_face_and_crop(img_arr)
        if not Image.isImageType(cropped_img): # Just if some images are broken or not found
            continue
        predicted_age, predicted_age_class, prob = predict_age(cropped_img)
        # Checking if predicted age is not in interval:
        if predicted_age in out_of_interval_years:
            out_of_years += 1
            continue
        exact = exact_accuracy(min,max,predicted_age)
        one_off = one_off_accuracy(predicted_age,label)
        if exact:
            exact_hits += 1
        if one_off:
            one_off_hits += 1
        n += 1
        if n % 100 == 0:
            print("-------------------------------------")
            print(f"Number of samples predicted as of now: {n}")
            print(f"Exact accuracy as of now: {np.round(exact_hits / n, 3)}")
            print(f"One-off accuracy as of now: {np.round(one_off_hits / n, 3)}")
            print("-------------------------------------","\n")

    print("FINISHED!!!")
    print(f"Number of samples predicted: {n}")
    print(f"Exact accuracy: {np.round(exact_hits/n,3)}")
    print(f"One-off accuracy: {np.round(one_off_hits/n,3)}")
    print(f"Number of out-of-bounds-predictions: {out_of_years}")

#create_fold_faces_folder()
run()


