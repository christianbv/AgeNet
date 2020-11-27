# Folder structure:
Note that all jupyter notebooks can be runned in Google Colaboratory. 

## Raw data:
- UTK data: https://susanqq.github.io/UTKFace/
- APPA data: http://chalearnlap.cvc.uab.es/dataset/26/description/
- IMDB-WIKI data: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

Note that both UTK and APPA is found in our bucket at: https://storage.googleapis.com/ntnu-ml-bucket/, whereas IMDB must be downloaded and used locally. 

## Preprocessing:
Three notebooks here. First two were used to detect faces and crop images: 

-> 0_Preprocessing_APPA_and_UTK_datasets.ipynb and 0_Preprocessing_IMDB_images.ipynb. 

The final was used to filter out mislabellings in the IMDB-WIKI dataset:
-> 1_Validate_IMDB_images.ipynb

## Training of model:
- 2_Train_model_single_and_multiple.ipynb

## Evaluation on test set:
- 3_evaluate_model_logs.ipynb

## Evaluation of Adience benchmark:
- Check out the Adience folder. The Adience benchmark is found at: https://talhassner.github.io/home/projects/Adience/Adience-data.html, where by creating a user one can download the files in its entirety. The file Adience/adience_validation.py can be used directly on the downloaded data. 

## Third-party implementations used:
- TensorFlow v/2.3.0
- Google Storage Bucket
- Google Colaboratory
- Face_recognition module from dlib
