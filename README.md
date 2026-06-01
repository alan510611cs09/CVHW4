# CVHW4

NYCU Computer Vision 2026 HW4

Student ID: 109550026

Name: 楊詠翔

# Introduction
To perform image restoration using PromptIR.


# Environment Setup
The default running environment is **Google Colab**. Using **G4** GPU.


If running on Colab, only lightning requires additionally installed.

`
pip install lightning
`

Otherwise, use the command to install packages needed:

`
pip install -r requirements.txt
`

# Usage
## Training data
If using Colab, the training data should be upload to cloud drive, and unzip it into Colab VM.

And download the official PromptIR repository.

<img width="688" height="270" alt="image" src="https://github.com/user-attachments/assets/73f70a42-5f4c-4241-8f76-853a92530cb9" />

## Replace Files
Replace those files to our custom ones.

1. **train.py** and **demo.py** under PromptIR/
2. **dataset_utils.py** under PromptIR/utils/
   
## Data management
After uploading dataset, we need to modify it to specific format for PromptIR training.

<img width="700" height="331" alt="image" src="https://github.com/user-attachments/assets/bb1043e6-d564-49b4-99e6-3b4e6f5d751a" />

## Training 
Press the start button to start training model.

<img width="597" height="325" alt="image" src="https://github.com/user-attachments/assets/9f562792-d762-4455-be1b-085f1d27690b" />



## Inference
After training is done, press the start button to start inference.

<img width="775" height="192" alt="image" src="https://github.com/user-attachments/assets/72024836-5bcf-4d4a-a16b-4d04a1cd12da" />


## Generate .npz file
After inference, convert 100 cleansed images to npz format.

<img width="438" height="257" alt="image" src="https://github.com/user-attachments/assets/9f4b0abb-ba26-491c-8223-49f8453d7bfb" />


# Performance Snapshot.

<img width="1299" height="53" alt="image" src="https://github.com/user-attachments/assets/1bf3b55d-602e-4ff5-a5a1-737d9e625a23" />
