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
from jnius import autoclass
from PIL import Image
import matplotlib.pyplot as plt
import _thread
import tkinter
import PySimpleGUI as sg

BluetoothAdapter = autoclass('android.bluetooth.BluetoothAdapter')
BluetoothDevice = autoclass('android.bluetooth.BluetoothDevice')
BluetoothSocket = autoclass('android.bluetooth.BluetoothSocket')
UUID = autoclass('java.util.UUID')

mixer.init()
#sound = mixer.music.load('nomask.mp3')
#sound1 = mixer.music.load('mask.mp3')

def get_socket_stream(name):
    paired_devices = BluetoothAdapter.getDefaultAdapter().getBondedDevices().toArray()
    #for i in paired_devices:
    	#print(i.getName(), i.getUuids()[0].toString())
    socket = None
    for device in paired_devices:
        if device.getName() == name:
            socket = device.createRfcommSocketToServiceRecord(
                UUID.fromString("00001101-0000-1000-8000-00805f9b34fb"))
            recv_stream = socket.getInputStream()
            send_stream = socket.getOutputStream()
            break
            
    socket.connect()
    return recv_stream, send_stream
    
   
#recv_stream, send_stream = get_socket_stream('HC-05')

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
	# locs = []
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
			#locs.append((startX, startY, endX, endY))

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

vs = VideoStream(src=0).start()
time.sleep(2.0)

def pl():
	window = sg.Window('Demo Application - OpenCV Integration', [[sg.Image(filename='', key='image')], ], location=(0, 0), grab_anywhere=True, margins=(0,400))
	
	while window(timeout=20)[0] is not None:
		mir = imutils.resize(vs.read(), width=1000, height=1500)
		window['image'](data=cv2.imencode('.png',mir)[1].tobytes())

maskTrue = 0
withoutMaskTrue = 0

# loop over the frames from the video stream
_thread.start_new_thread(pl, ())

while True:
	
	try:
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		
		(preds) = detect_and_predict_mask(frame, faceNet, maskNet)
		
		for pred in preds:
	        #    (startX, startY, endX, endY) = box
			(mask, withoutMask) = pred
			
			if mask > 0.90:
				maskTrue += 1
			else:
				withoutMaskTrue += 1
	           	
	        #    label = "Mask" if mask > 0.90 else "No Mask"
	        #    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
	        #    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
	        #    cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
	        #    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	except:
		pass
	
	if maskTrue == 2:
		maskTrue = 0
		withoutMaskTrue = 0
		mixer.music.load('mask.mp3')
		mixer.music.play()
		s="relay2 on"
		#send_stream.write(s.encode('ascii'))
		time.sleep(5.0)
		#send_stream.flush()
		#p="relay3 on"
		#send_stream.write(p.encode('ascii'))
		#send_stream.flush()
			
	if withoutMaskTrue == 2:
		withoutMaskTrue = 0
		maskTrue = 0
		mixer.music.load('nomask.mp3')
		mixer.music.play()
		time.sleep(5.0)

	# show the output frame
	#cv2.imshow("Frame", frame)

	
	# key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	# if key == ord("q"):
		# break

# do a bit of cleanup
#cv2.destroyAllWindows()
vs.stop()