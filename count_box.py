import os
#預測影像，並將結果存成txt
os.system('./darknet detector test data/obj.data cfg/custom-yolov4-detector.cfg backup/custom-yolov4-detector_last.weights -dont_show -ext_output dataset/test/3.jpg > result.txt')

#將結果存成陣列，共6個資訊，
#類別名稱、信心度、left_x、top_y、width、height
import re
detections = []

with open('result.txt') as f:
  for line in f:
      predict_information = re.findall(r'[-\d]+', line)
      if "box:" in line:
          detections.append(['box'] + predict_information)
      if "box_top:" in line:
          detections.append(['box_top'] + predict_information)
#print(detections)

import cv2
import matplotlib.pyplot as plt
import numpy as np
box_top_s = 150 #堆疊高度(超過代表桶槽有堆疊)
box_space = 2 #區域個數
box_line = [850, 1400] #區域分割線，由左到右排序(X軸)
box_bottom = 650 #區域底線(Y軸)
def box_count(detections, box_top_s, box_space, box_line, box_bottom):
  box_num = np.zeros((box_space))
  y_top_list = [[] for i in range(box_space)]
  y_top_box_list = [[] for i in range(box_space)]
  for detection in detections:
    label = detection[0]
    box_height = int(detection[5])
    box_width = int(detection[4])
    # 計算 Box 座標
    x_left = int(detection[2])
    y_top = int(detection[3])
	#過濾目標區域以外的桶子
    if x_left + box_width > box_line[box_space - 1] or y_top > box_bottom: # or confidence < 0.9
      continue
	#根據區域數量，單獨儲存box_top和box的y_top
    for i in range(box_space):
      if label == "box_top":
        if x_left + box_width <= box_line[i]:
          y_top_list[i].append(y_top)
          box_num[i] = box_num[i] + 1
          break
      elif label == "box":
        if x_left + box_width <= box_line[i]:
          y_top_box_list[i].append(y_top)
          break
  #list to numpy array
  for i in range(box_space):
    y_top_list[i] = np.array(y_top_list[i])
    y_top_box_list[i] = np.array(y_top_box_list[i])
  #處理桶子堆疊問題
  for i in range(box_space):
    #判斷區域內是否有桶子
    if y_top_list[i].size != 0:
      diff_num = y_top_list[i][(y_top_list[i] - y_top_list[i].min()) > box_top_s] #第一層一噸桶
	  #計算桶子數量
	  #計算流程：將第一層的桶子過濾掉，將剩下的桶子數量乘以層高，再加上第一層的桶子數量
      if diff_num.size != 0:
        box_num[i] = (box_num[i] - diff_num.size) * 2 + diff_num.size
      else:
	  #box_top都在同一層時，將box_top數量乘以層高，即一噸桶數量
        if y_top_list[i].size != 0:
          if (y_top_box_list[i].max() - y_top_box_list[i].min()) > box_top_s:
            box_num[i] = box_num[i] * 2
  return box_num

box_num = box_count(detections, box_top_s, box_space, box_line, box_bottom)

print("Box_1:",box_num[0])
print("Box_2:",box_num[1])
print("Area_1:",str(int((box_num[0] / 15.0) * 100)) + "%")
print("Area_2:",str(int((box_num[1] / 14.0) * 100)) + "%")

#將結果顯示在影像中
'''
imagePath = "box_image/151120.jpg"
image = cv2.imread(imagePath)
image_o = image.copy()
for detection in detections:
  label = detection[0]
  confidence = detection[1]
  pstring = label + ": " + str(confidence) + "%"
  box_height = int(detection[5])
  box_width = int(detection[4])
  # 計算 Box 座標
  x_left = int(detection[2])
  y_top = int(detection[3])
  if x_left + box_width > box_line[box_space - 1] or y_top > box_bottom:
    continue
  # 在影像中標出Box邊界和類別、信心度
  if label == "box":
    rectColor = (0, 255, 0)
    textCoord = (x_left, y_top - 10)
  elif label == "box_top":
    rectColor = (0, 255, 255)
    textCoord = (x_left, y_top + 20)
	
  cv2.rectangle(image, (x_left, y_top), (x_left + box_width, y_top + box_height), rectColor, 2)
  cv2.putText(image, pstring, textCoord, cv2.FONT_HERSHEY_DUPLEX, 1, rectColor, 2)

blue = (255, 0, 0)
#標出各類別數量
cv2.putText(image, "Box_1: " + str(int(box_num[0])), (10,80), cv2.FONT_HERSHEY_DUPLEX, 2, blue, 2)
cv2.putText(image, "Box_2: " + str(int(box_num[1])), (10,160), cv2.FONT_HERSHEY_DUPLEX, 2, blue, 2)
cv2.imwrite("box_num.jpg", image)
fig=plt.figure(figsize=(16, 12))
fig.add_subplot(2, 1, 1)
plt.imshow(cv2.cvtColor(image_o, cv2.COLOR_BGR2RGB))
fig.add_subplot(2, 1, 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
'''
