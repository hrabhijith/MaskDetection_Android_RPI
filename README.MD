# Face Mask Detection using MobileNet pretrained Model (Android and Raspberry Pi)

This program or software detects the human face from the camera feed (Either from Android camera, Raspberry pi camera or Any other camera source if configured) and then properly detects whether the face has mask on it with mouth and nose covered properly as defined by WHO. 

This software has three main parts
1. The training python code and dataset (Training the MobileNet Model with dataset)
2. Detection python code for Android
3. Detection python code for Raspberry Pi (Taking camera feed and giving inputs and fetching outputs from trained model)

## Installation

### Dataset

The dataset folder as two sub folders. One for storing human faces with mask and another for without mask. Here, only one image is saved in each folders as examples (My dataset had around 2k images, cannot be stored in GitHub, you can collect your own images)

### Training code and dataset (PC)

Open the terminal in any of your code editor in PC and open the train_mask_detector.py

In the terminal run the following commands.

** pip3 install -r requirements.txt **

Then run the train_mask_detector.py file after placing proper images in the dataset folder.

The output is a model file and loss/accuracy plot. Model file is further used in detector python files.

### Android

Python coding is made possible in Android using Pydroid 3 android app which can found in Playstore.
After installing the Pydroid (Python 3 will be automatically accessed), open this project and using terminal run following commands

** pip3 install -r requirements.txt **

Then run the detect_mask_android.py file.

### Raspberry Pi

After installing proper OS (Raspbian preffered) and connecting either RPI camera or USB camera (USB camera needs little change in python detector code for rpi) you can open this project in RPI (Any code editor) and in the terminal run the following command

** pip3 install -r requirements.txt **

Then run the detect_mask_rpi.py file.

Note: RPI needs to be installed with few lib modules. Check google on how to run tensorflow and opencv in RPI.