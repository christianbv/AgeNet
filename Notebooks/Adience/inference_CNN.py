import cv2
import numpy as np
from keras import models
import tensorflow as tf
from tensorflow import keras
import sys
from PIL import Image
import face_recognition

# Initialize some variables
face_locations = []
face_ages = []

video = cv2.VideoCapture(0)
process_this_frame = True
model_path = "/Users/christianbv/PycharmProjects/ML/venv/models_model_adam_v2_14_11"

# Loading the model for inference
model = models.load_model(model_path)
class_labels = ['0-2', '3-6', '7-12', '13-17', '18-22', '23-26', '27-33', '34-44', '45-59', '60-199']
print("Model loaded...", '\n','\n')


"""
    Converts the multi-class predictions into one single value
    Using the average value of each bucket here - naturally not very precise, however it gives an indication
"""
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
    The logic which runs everything
    Press q for exit
"""
while True:
    j = 0
    _, frame = video.read()
    # Gjør bildet mindre
    smaller_img = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)
    # Endrer fra BGR til RGB (face recognition)
    rgb_smaller = smaller_img[:,:,::-1]

    # Prossesserer bare annenhvert bilde:
    if j % 4 == 0:
        # Henter ut lokasjoner og encodings for hvert fjes i denne video framen:
        face_locations = face_recognition.face_locations(rgb_smaller)
        im = Image.fromarray(rgb_smaller)
        images = []

        for (top,right,bottom,left) in face_locations:
            padding = abs(left-right)*0.4
            im_cropped = im.crop((left - padding, top - padding, right + padding, bottom + padding))
            im_resized = im_cropped.resize((256,256), Image.ANTIALIAS)
            images.append(im_resized)

        # Predikerer alder:
        face_ages = []
        for image in images:
            age_acc, age_pred, prob = predict_age(image)
            text = f'Age: {age_acc}. {np.round(prob*100,3)}%'
            #text2 = f'Age: {age_acc}, interval: {age_pred}. {np.round(prob*100,3)}%'       # Uncomment for both regressor and class with highest score
            face_ages.append(text)

    j = 0

    # Display the results
    for (top, right, bottom, left), age in zip(face_locations, face_ages):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, age, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
