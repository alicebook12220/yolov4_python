import cv2
import numpy as np
import time
import glob

net = cv2.dnn_DetectionModel('cfg/custom-yolov4-tiny.cfg', 'backup/custom-yolov4-tiny_best.weights')
net.setInputSize(608, 608)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)


imagePath = "D:\\harrylin\\project_list\\AVMap\\Image\\dataset\\DEFECT_Y\\*.jpg" 
imagePath = glob.glob(imagePath)
imagePath.sort()

for i, path in enumerate(imagePath):
	if i % 100 == 0:
		print("processing " + str(i) + " images")
	image = cv2.imread(path)
	img_name = path.split("\\")
	img_name = img_name[len(img_name) - 1]
	img_name = img_name.split(".")
	img_name = img_name[0]
	#start = time.time()
	classes, confidences, boxes = net.detect(image, confThreshold=0.7, nmsThreshold=0.5) #0.9s
	#end = time.time()
	#print(end - start)
	if len(boxes) > 0:
		for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
			pstring = str(int(100 * confidence)) + "%" #信心度
			x_left, y_top, width, height = box

			boundingBox = [
				(x_left, y_top), #左上頂點
				(x_left, y_top + height), #左下頂點
				(x_left + width, y_top + height), #右下頂點
				(x_left + width, y_top) #右上頂點
			]
			'''
			if classId == 0:
				rectColor = (255, 0, 0)
				textCoord = (x_left, y_top - 10)
			'''
			rectColor = (255, 0, 0)
			textCoord = (x_left, y_top - 10) #文字位置
			# 在影像中標出Box邊界和類別、信心度
			cv2.rectangle(image, boundingBox[0], boundingBox[2], rectColor, 2)
			cv2.putText(image, pstring, textCoord, cv2.FONT_HERSHEY_DUPLEX, 1, rectColor, 2)
			cv2.imwrite("predict/" + img_name + "_" + pstring + ".jpg", image)
	else:
		cv2.imwrite("nothing/" + img_name + "_predict.jpg", image)
	