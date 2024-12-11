# Neural Network Final Project: ASL Alphabet Recognition (using CNN and ResNet)
Authors: Kendall Gilbert, Hannah Kim

What is Included in Repo:
- CNN Model: main_cnn.ipynb
- ResNet Model: resnet.ipynb
- First CNN Model attempted using old data: firstmodel.ipynb
- Saved Model Folder: Models
- Streamlit Application: app.py
- Report: 6600 Final Project Report.pdf

## How to run the CNN model with Augmented Data
run

`streamlit run app.py`

on the terminal under the root directory

## How to run the ResNet model with Augmented Data

- go to app.py file
- change the model path to __'./models/resnet_model.h5'__
- run `streamlit run app.py`

## How to run the Augmented model with original Kaggle Data

- go to app.py file
- change the model path to __'./models/augmented_model.h5'__
- run `streamlit run app.py`

