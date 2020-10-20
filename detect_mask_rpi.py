from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from pygame import mixer
from videostream import VideoStream
import _thread
from PIL import Image
#import RPi.GPIO as GPIO

mixer.init()

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			# locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		preds = maskNet.predict(faces)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return preds

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")

vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# GPIO.setmode(GPIO.BCM)

# GPIO.setup(16, GPIO.OUT)
# GPIO.setup(20, GPIO.OUT)
# GPIO.setup(6,GPIO.IN)
# GPIO.setup(5,GPIO.OUT)
# GPIO.setup(27, GPIO.IN)
# GPIO.setup(22, GPIO.IN)



def live():
    global vs
    while True:
        frame1 = vs.read()
        frame1 = imutils.resize(frame1, width=1360, height=600)
        cv2.imshow('Frame', frame1)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    
    cv2.destroyAllWindows()
    vs.stop()
    GPIO.cleanup()
        
_thread.start_new_thread(live,())

maskTrue = 0
withoutMaskTrue = 0

while True:
	mixer.music.load('start.mp3')
	mixer.music.play()
	time.sleep(5.0)
	mixer.music.load('stand.mp3')
	mixer.music.play()
	time.sleep(2.0)
		
	while True:
		try:
			frame = vs.read()
			frame = imutils.resize(frame, width=400)
			preds = detect_and_predict_mask(frame, faceNet, maskNet)
				
			for pred in preds:
				(mask, withoutMask) = pred
				if mask > 0.90:
					maskTrue += 1
				else:
					withoutMaskTrue += 1
		except:
			pass
			
		if maskTrue == 2:
			maskTrue = 0
			withoutMaskTrue = 0
			mixer.music.load('mask.mp3')
			mixer.music.play()
			time.sleep(5.0)
			print('mask')
				
		if withoutMaskTrue == 2:
			withoutMaskTrue = 0
			maskTrue = 0
			mixer.music.load('nomask.mp3')
			mixer.music.play()
			time.sleep(5.0)
			print('no mask')

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()