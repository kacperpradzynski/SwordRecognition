import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-c", "--config", required=True,
	help="base path to YOLO directory")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-t", "--threshold", type=float, default=0.4,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

videoOutput = args["output"]

# load class labels
labelsPath = os.path.sep.join([args["config"], "classes.names"])
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = [[255, 255, 51], [255, 51, 255]]

weightsPath = os.path.sep.join([args["config"], "yolov3.weights"])
configPath = os.path.sep.join([args["config"], "yolov3.cfg"])

# load YOLO object detector
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

while True:
	(grabbed, frame) = vs.read()

	if not grabbed:
		break

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	starts = []
	ends = []
	startX = "#"
	startY = "#"
	endX = "#"
	endY = "#"

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > args["threshold"]:
				# scale the bounding box coordinates
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# update list of bounding boxes coordinates
				if classID==0:
					starts.append([centerX, centerY, classID, float(confidence)])
				else:
					ends.append([centerX, centerY, classID, float(confidence)])

	if len(starts) > 0:
		maxStart = starts[np.argmax(np.vstack(starts)[:,3])]
		(x, y) = (maxStart[0], maxStart[1])
		startX = x
		startY = y
		if videoOutput is not None:
			color = [int(c) for c in COLORS[maxStart[2]]]
			cv2.circle(frame, (x, y), 1, color, 5)

	if len(ends) > 0:
		maxEnd = ends[np.argmax(np.vstack(ends)[:,3])]
		(x, y) = (maxEnd[0], maxEnd[1])
		endX = x
		endY = y
		if videoOutput is not None:
			color = [int(c) for c in COLORS[maxEnd[2]]]
			cv2.circle(frame, (x, y), 1, color, 5)

	if videoOutput is not None:
		if writer is None:
			# initialize video writer
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			writer = cv2.VideoWriter(args["output"], fourcc, 24,
				(frame.shape[1], frame.shape[0]), True)
		# write the output frame to disk
		writer.write(frame)

	print(f'{startX};{startY};{endX};{endY}')

# release the file pointers
if videoOutput is not None:
	writer.release()
vs.release()