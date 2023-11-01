from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as keras_preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array as keras_img_to_array
from tensorflow.keras.models import load_model as keras_load_model
from imutils.video import VideoStream as imutils_VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def MAIN_FUNCTION_FOR_DETECTION(frame, face_net, mask_net):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
	face_net.setInput(blob)
	detections = face_net.forward()
	print(detections.shape)
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = keras_img_to_array(face)
			face = keras_preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = mask_net.predict(faces, batch_size=32)
	return (locs, preds)

prototxt_path = "Main.prototxt"
weights_path = "Main2.caffemodel"
face_net = cv2.dnn.readNet(prototxt_path, weights_path)
mask_net = keras_load_model("Detector.model")
print("[INFO] starting video stream...")
vs = imutils_VideoStream(src=0).start()
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=1100)
	(locs, preds) = MAIN_FUNCTION_FOR_DETECTION(frame, face_net, mask_net)
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, without_mask) = pred
		label = "Mask" if mask > without_mask else "No Mask"
		color = (255,5,5) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()
