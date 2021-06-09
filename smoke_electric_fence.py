import cv2
import numpy as np
import time
import glob

def grid_exist_top(box, grid_y = 180, grid_x_left = 0, grid_x_right = 135):
	x_left, y_top, width, height = box
	x_right = x_left + width
	if y_top < grid_y:
		if x_left >= grid_x_left and x_left <= grid_x_right:
			if grid_x_right - x_left > (width / 2):
				return "1"
		elif x_right >= grid_x_left and x_right <= grid_x_right:
			if x_right - grid_x_left > (width / 2):
				return "1"
	return "0"
	
def grid_exist_down(box, grid_y = 705, road_y = 750, grid_x_left = 190, grid_x_right = 430):
	x_left, y_top, width, height = box
	y_down = y_top + height
	x_right = x_left + width
	if y_down > grid_y and y_down < road_y:
		if x_left >= grid_x_left and x_left <= grid_x_right:
			if grid_x_right - x_left > (width / 2):
				return "1"
		elif x_right >= grid_x_left and x_right <= grid_x_right:
			if x_right - grid_x_left > (width / 2):
				return "1"
	return "0"
	
	
net = cv2.dnn_DetectionModel('cfg/enet-coco.cfg', 'model/enetb0-coco_final.weights')
#net = cv2.dnn_DetectionModel('cfg/yolov4-tiny.cfg', 'model/yolov4-tiny.weights')
net.setInputSize(416, 416)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)

#net = cv2.dnn.readNet('custom-yolov4-detector_last.weights', 'custom-yolov4-detector.cfg')
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
#model = cv2.dnn_DetectionModel(net)
#model.setInputParams(size = (512, 512), scale = 1.0/255)

#imagePath = "D:\\harrylin\\AVMap\\darknet\\608\\dataset\\test\\*.jpg" 
#imagePath = "D:\\harrylin\\AVMap\\Image\\dataset\\DEFECT_N\\*.jpg" 
imagePath = "0604\\image-4\\1.jpg" 
imagePath = glob.glob(imagePath)
print(len(imagePath))
imagePath.sort()

for i, path in enumerate(imagePath):
	grid_43 = "0"
	grid_44 = "0"
	grid_45 = "0"
	grid_46 = "0"
	grid_47 = "0"
	grid_48 = "0"
	grid_12 = "0"
	grid_13 = "0"
	grid_14 = "0"

	image = cv2.imread(path)
	img_name = path.split("\\")
	img_name = img_name[len(img_name) - 1]
	img_name = img_name.split(".")
	img_name = img_name[0]
	#start = time.time()
	classes, confidences, boxes = net.detect(image, confThreshold=0.0, nmsThreshold=0.5) #0.1s
	#end = time.time()
	#print(end - start)
	start = time.time()
	if len(boxes) > 0:
		for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
			if classId != 0:
				continue
			if grid_43 == "0":
				grid_43 = grid_exist_top(box, grid_y = 180, grid_x_left = 0, grid_x_right = 135)
			if grid_44 == "0":
				grid_44 = grid_exist_top(box, grid_y = 180, grid_x_left = 136, grid_x_right = 315)
			if grid_45 == "0":
				grid_45 = grid_exist_top(box, grid_y = 180, grid_x_left = 316, grid_x_right = 500)
			if grid_46 == "0":
				grid_46 = grid_exist_top(box, grid_y = 180, grid_x_left = 501, grid_x_right = 672)
			if grid_47 == "0":
				grid_47 = grid_exist_top(box, grid_y = 180, grid_x_left = 673, grid_x_right = 847)
			if grid_48 == "0":
				grid_48 = grid_exist_top(box, grid_y = 180, grid_x_left = 848, grid_x_right = 1019)
			if grid_12 == "0":
				grid_12 = grid_exist_down(box, grid_y = 705, road_y = 750, grid_x_left = 235, grid_x_right = 437)
			if grid_13 == "0":
				grid_13 = grid_exist_down(box, grid_y = 705, road_y = 750, grid_x_left = 438, grid_x_right = 656)
			if grid_14 == "0":
				grid_14 = grid_exist_down(box, grid_y = 705, road_y = 750, grid_x_left = 657, grid_x_right = 828)
			
		result = "43:" + grid_43 + " " + "44:" + grid_44 + " " + "45:" + grid_45 + " " + "46:" + grid_46 + " " + "47:" + grid_47 + " " + "48:" + grid_48 + " " + "12:" + grid_12 + " " + "13:" + grid_13 + " " + "14:" + grid_14
		print(result)
	end = time.time()
	print(end - start)