# AgeNet
This project is the final assignment in TDT4173 Machine Learning, NTNU Trondheim. Adrian Kjærran, Erling Stray-Bugge and Christian Vennerød have created a deep CNN network for multi-class age classification. The notebooks are published here in this repository, along with a script which you can download and run for yourselves. Please note that the script is a hybrid variant, where a exact age is estimated from the predicted softmax class distributions and an assumption that mean ages in each class is the middle age, e.g. mean age of (30-35) is 32.5. 

## Link to YouTube video:
- Our YouTube video is here: https://www.youtube.com/watch?v=haNNlZm7L2o&ab_channel=ChrisBv

## Estimate your own age locally

Only tested on MacOS Catalina with Python 3.8.   
Required dependencies: TensorFlow, Keras, CV2, face_recognition, Numpy, and PIL.

1. Download the file 'inference_CNN.py' from this repo
2. Download our TensorFlow age estimation models:
  - Either our ADAM model, from https://storage.cloud.google.com/ntnu-ml-bucket/models/model_adam_v2_14_11.zip (authenticate with Google)
  - Our our CNN model, from https://storage.cloud.google.com/ntnu-ml-bucket/models/cnn_20201114_143814.zip (authenticate with Google). 
3. Unzip the zip file, and place the folder in a convenient destination. 
4. Copy the absolute destination path for the folder, and replace the path at line 16 under the variable "model_path" with your own path, in string format. 
5. Run the model, and estimate your own age :D 

## Notebooks
Contains:

1. Code for downloading the different datasets
2. Preprocessing the datasets
3. Training the models 
4. Evaluating the models by visualizing metrics in Tensorboard.

Please check out the READme file in the notebooks folder for further descriptions. 

## Datasets:
- Remember to authenticate with Google. 

Link to Appa dataset: https://storage.cloud.google.com/ntnu-ml-bucket/Appa/appa_by_age_256_FIXED.zip
Link to IMDB dataset: https://storage.cloud.google.com/ntnu-ml-bucket/IMDB/IMDB_by_age.zip
Link to UTK dataset:  https://storage.cloud.google.com/ntnu-ml-bucket/Utk/utk_by_age_256_FIXED.zip

## Finished trained models: 
- Remember to authenticate with Google.

1. A deep-CNN with 5 conv. layers and adam was trained on our data. This is our final model, and is found here:
- https://storage.cloud.google.com/ntnu-ml-bucket/models/model_adam_v2_14_11.zip  

2. When training this model, we logged to GSB for each batch. The final zip file with logs from each batch when training our DCNN is found here (reproducibility):
- https://storage.cloud.google.com/ntnu-ml-bucket/logs/logg-num-logs-10-session-8-id-1793.zip. 

