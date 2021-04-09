import pyautogui
import pyperclip
import cv2
import numpy as np
import time
import glob

def img2cv(im, gray = 1):
	im = np.array(im)
	im = im[:, :, ::-1].copy()
	if gray == 0:
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	return im

net = cv2.dnn_DetectionModel('cfg/custom-yolov4-tiny.cfg', 'backup/custom-yolov4-tiny_last.weights')
net.setInputSize(1024, 768)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)

#net = cv2.dnn.readNet('custom-yolov4-detector_last.weights', 'custom-yolov4-detector.cfg')
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
#model = cv2.dnn_DetectionModel(net)
#model.setInputParams(size = (512, 512), scale = 1.0/255)

#imagePath = "D:\\harrylin\\AVMap\\darknet\\608\\dataset\\test\\*.jpg" 
#imagePath = "D:\\harrylin\\AVMap\\Image\\dataset\\DEFECT_N\\*.jpg" 
#imagePath = "C:\\Users\\HarryLin\\Documents\\darknet_win_test\\AVMap\\crack_point\\66\\*jpg" 
imagePath = "D:\\harrylin\\project_list\\crack_point\\crack\\2021-04-01\\2021-04-01_9_crack\\1\\*jpg" 


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
	classes, confidences, boxes = net.detect(image, confThreshold=0.1, nmsThreshold=0.5) #0.9s
	#end = time.time()
	#print(end - start)
	if len(boxes) > 0:
		for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
			pstring = str(int(100 * confidence)) + "%"
			x_left, y_top, width, height = box
	#		x_center = int(x_left + (width/2)) + 205
	#		y_center = int(y_top + (height/2)) + 90
	#		pyautogui.moveTo(x_center, y_center, duration=0.25)
	#		time.sleep(0.5)
	#		pyautogui.click()
	#		time.sleep(0.5)
			boundingBox = [
				(x_left, y_top), #左上頂點
				(x_left, y_top + height), #左下頂點
				(x_left + width, y_top + height), #右下頂點
				(x_left + width, y_top) #右上頂點
			]
			if classId == 0:
				rectColor = (0, 0, 255)
				textCoord = (x_left, y_top - 10)
			if int(100 * confidence) > 1:
			# 在影像中標出Box邊界和類別、信心度
				cv2.rectangle(image, boundingBox[0], boundingBox[2], rectColor, 2)
				cv2.putText(image, pstring, textCoord, cv2.FONT_HERSHEY_DUPLEX, 1, rectColor, 2)
				cv2.imwrite("output/Y/" + img_name + "_" + pstring + "_predict.jpg", image)
	else:
		cv2.imwrite("output/N/" + img_name + "_predict.jpg", image)
	
#time.sleep(5)
#image = pyautogui.screenshot()
#image = img2cv(image)
#image = image[90:759, 205:888]