from ctypes import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse
import sys
import glob
import os 
import time
import datetime
import threading

class METADATA(Structure):
    _fields_ = [("classes", c_int),
          ("names", POINTER(c_char_p))]
class IMAGE(Structure):
    _fields_ = [("w", c_int),
          ("h", c_int),
          ("c", c_int),
          ("data", POINTER(c_float))]
class BOX(Structure):
    _fields_ = [("x", c_float),
          ("y", c_float),
          ("w", c_float),
          ("h", c_float)]
class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
          ("classes", c_int),
          ("prob", POINTER(c_float)),
          ("mask", POINTER(c_float)),
          ("objectness", c_float),
          ("sort_class", c_int),
          ("uc", POINTER(c_float)),
          ("points", c_int),
          ("embeddings", POINTER(c_float)),
          ("embedding_size", c_int),
          ("sim", c_float),
          ("track_id", c_int)]

lib = CDLL("C:/Users/2007041/Desktop/darknet-master/build/darknet/x64/yolo_cpp_dll_nogpu.dll", RTLD_GLOBAL)
load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p
load_meta = lib.get_metadata
load_meta.argtypes = [c_char_p]
load_meta.restype = METADATA
load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE
predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)
get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)
do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]
free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection
    """
    predictions = []
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], (bbox)))
    return predictions

def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
	"""
		Returns a list with highest confidence class and their bbox
	"""
	pnum = pointer(c_int(0))
	predict_image(network, image)
    
	detections = get_network_boxes(network, image.w, image.h, thresh, hier_thresh, None, 0, pnum, 0)
    
	num = pnum[0]
	if nms:
		do_nms_sort(detections, num, len(class_names), nms)

	predictions = remove_negatives(detections, class_names, num )
	free_detections(detections, num)
	return sorted(predictions, key=lambda x:-x[1])
		
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath , dtype = np.uint8),-1)
    return cv_img


#判斷是否重疊
def Intersecting_area(box1x,box1y,box1w,box1h,box2x,box2y,box2w,box2h):
	#p1為相交位置的左上角坐標，p2為相交位置的右下角坐標
	p1x = max(box1x,box2x) 
	p1y = max(box1y,box2y) 
	p2x = min(box1x+box1w , box2x+box2w)
	p2y = min(box1y+box1h , box2y+box2h)

	if p2x > p1x and p2y > p1y :          #判斷是否相交，有重疊回傳1，無重疊回傳0
		return 1;
		                
	else:
		return 0;	

		
configPath = "C:/Users/2007041/Desktop/ALCV/custom-yolov4-detector.cfg"
weightPath = "C:/Users/2007041/Desktop/ALCV/custom-yolov4-detector_best.weights"
metaPath = "C:/Users/2007041/Desktop/ALCV/obj.data"

network = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
metadata = load_meta(metaPath.encode("ascii"))
class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]

#-----------------------------尋找修補的pixel範圍-------------------------------------
def job():
	global jpg_file
	global 	xc
	global	yc
	global	finalw 
	global	finalh 
	#global  draw_img
	global  img

	
	#img = cv2.resize(img, (1056, 783), interpolation=cv2.INTER_CUBIC)
    #轉灰階圖
	img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #自適應二值化                  
	th1 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,10)
	#形態學 erosion
	kernel = np.ones((5,5),np.uint8) 
	#element = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
	erosion = cv2.erode(th1,kernel,iterations = 3)
	#erosion後 findcounter 
	contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)  

    #水平投影
	pr = th1.copy()
	h = img.shape[0]
	w = img.shape[1]
	#圖片中心點座標
	centerh = (h+1)//2
	centerw = (w+1)//2
	a=[0 for z in range(0,h)]
	for i in range(0,h):
		for j in range(0,w):
			if pr[i,j] == 0:
				a[i]+=1
				pr[i,j]=255

	for i in range(0,h):
		for j in range(0,a[i]):
			pr[i,j]=0  
    #y方向影像切割
	asum = 0
	for i in range(0,len(a)):
		asum = asum + a[i]

	count1 = h//2
	amean = asum//h
	hflag = False
    #從中心開始往上下找，若先遇到黑色最多然後再遇到黑色低點就紀錄
	while count1 >0:    
		blackpoint = a[count1]
		if blackpoint > amean+50:
			hflag = True
		if hflag and  blackpoint < amean-5:  
			yf=count1
			break
		count1 =count1-1    
					
	count1 = h//2
	hflag = False
	while count1 <h: 
		blackpoint = a[count1]
		if blackpoint > amean+50:
			hflag = True
		if hflag and  blackpoint < amean-5:  
			yb=count1
			break
		count1 =count1+1

	
	wall =[]
	hall =[]
	wcenter=[]
	hcenter=[]
	wc=0
	hc=0
	#通過中心點的原矩形
	originx =0
	originy =0
	originw =0
	originh =0
	#中位數矩形增加的長寬
	addw = 25
	addh = 40
	for cnt in contours:
		if cv2.contourArea(cnt)>= 2500 and cv2.contourArea(cnt)<= 100000:
			x, y, w, h = cv2.boundingRect(cnt)
           #記錄在中心排範圍內的矩形
			if  (yf-30) <= y and y+h <= (yb+30):
				wall.append(w)
				hall.append(h)
            #紀錄通過圖片正中心的矩形    
			if y <= centerh <= (y+h) and x <= centerw <= (x+w):
				originx = x +(w//2)
				originy = y +(h//2)
				originh = h
				originw = w

    #中位數        
	wcenter = sorted(wall)
	hcenter = sorted(hall)
	wc = wcenter[(len(wcenter)//2)]
	hc = hcenter[(len(hcenter)//2)]
	
    #若找不到中心pixel矩形則直接以正中心為基準畫中位數矩形
	if (originx ==0) or (originy ==0):
		xc = centerw - ((wc+addw)//2)
		yc = centerh - ((hc+addh)//2)
		finalw = wc+addw
		finalh = hc+addh
	else:
		if (wc*1.5) <= originw <= (wc*2.5):
			if (hc) < (originh) < (hc*2):
				xc = originx-((originw+addw)//2)
				yc = originy-((originh+addh)//2)
				finalw = originw + addw
				finalh = originh + addh
			else:
				xc = originx-((originw+addw)//2)
				yc = originy-((hc+addh)//2)
				finalw = originw + addw
				finalh = hc + addh
		else:    
			xc = originx-((wc+addw)//2)
			yc = originy-((hc+addh)//2)
			finalw = wc+addw
			finalh = hc+addh
	#cv2.rectangle(draw_img, (xc, yc), (xc + finalw ,yc + finalh ), (255 ,0, 255), 3)	




while(True):
	time.sleep(0.1)
	if os.listdir(os.fspath("C:/Users/2007041/Desktop/ALCV/test/")):   
		#time.sleep(0.1)
		f = glob.iglob(r"C:/Users/2007041/Desktop/ALCV/test/*.jpg")
		ww = "C://Users//2007041//Desktop//ALCV//test//"
		for jpg_file in f:
			if jpg_file:
				print("jpg name=",jpg_file)
				tStart = time.time()		#計時	
				txtfile_name = os.path.basename(jpg_file).replace(".jpg",".txt")
				file_name = os.path.basename(jpg_file)
				result = 0 		#是否要修補 ，0 = 可修 ，1 = ALCV不修
				labels = []
				confidence = []
				bounds = []
				img = cv_imread(jpg_file)
				#img = cv2.resize(img, (1056, 783), interpolation=cv2.INTER_CUBIC)
				#draw_img = img.copy()
				#-----------------------------尋找修補的pixel範圍-------------------------------------
				xc = 0
				yc = 0
				finalw = 0
				finalh = 0
				# 建立一個子執行緒
				t = threading.Thread(target = job)
				# 執行該子執行緒
				t.start()					
				#--------------------------------------YOLO辨識----------------------------------------
				im = load_image(jpg_file.encode("ascii"),0,0)
				detections = detect_image(network, class_names, im)
				for detection in detections :
					labels.append(detection[0])
					confidence.append(detection[1])
					bounds.extend(detection[2])
                

				# 等待 t 這個子執行緒結束
				t.join()
				
                #----------------------------------------判斷是否有ALCV在中心矩形----------------------                 
				count = 0                                   
				for label in labels:
					#print(label)
					if label == "ALCV":
						xEntent = int(bounds[count+2])     
						yEntent = int(bounds[count+3])
						xCoord = int(bounds[count] - bounds[count+2]/2)
						yCoord = int(bounds[count+1] - bounds[count+3]/2)
						count = count+4
						#判斷ALCV是否在中心pixel,1 = ALCV在中心pixel，0 = ALCV不在中心       
						Intersect = Intersecting_area(xc,yc,finalw,finalh,xCoord,yCoord,xEntent,yEntent)
						#cv2.rectangle(draw_img, (xCoord, yCoord), (xCoord + xEntent ,yCoord + yEntent ), (0,255, 255), 3)
						if Intersect == 1:
							result = 1
							break
						else:
							result = 0
                            
					else :
						result = 0
				#print(result)
				tEnd = time.time()		#計時
				process_time = tEnd - tStart  
				#cv2.imwrite("C:/Users/2007041/Desktop/ALCV/test_img/"+file_name,draw_img)
		        #讀取系統時間
				ISOTIMEFORMAT = '%Y-%m-%d,%H:%M'
				theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
				jpgfile = os.path.splitext(file_name)
				ff = open(ww+txtfile_name,'w')
				seq = [jpgfile[0],"\t",theTime,"\t",str(result),"\t",str(process_time)]
				ff.writelines(seq)
				ff.close()
				os.remove(jpg_file)#刪除檔案







