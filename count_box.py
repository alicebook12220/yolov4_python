import os
#�w���v���A�ñN���G�s��txt
os.system('./darknet detector test data/obj.data cfg/custom-yolov4-detector.cfg backup/custom-yolov4-detector_last.weights -dont_show -ext_output dataset/test/3.jpg > result.txt')

#�N���G�s���}�C�A�@6�Ӹ�T�A
#���O�W�١B�H�߫סBleft_x�Btop_y�Bwidth�Bheight
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
box_top_s = 150 #���|����(�W�L�N���Ѧ����|)
box_space = 2 #�ϰ�Ӽ�
box_line = [850, 1400] #�ϰ���νu�A�ѥ���k�Ƨ�(X�b)
box_bottom = 650 #�ϰ쩳�u(Y�b)
def box_count(detections, box_top_s, box_space, box_line, box_bottom):
  box_num = np.zeros((box_space))
  y_top_list = [[] for i in range(box_space)]
  y_top_box_list = [[] for i in range(box_space)]
  for detection in detections:
    label = detection[0]
    box_height = int(detection[5])
    box_width = int(detection[4])
    # �p�� Box �y��
    x_left = int(detection[2])
    y_top = int(detection[3])
	#�L�o�ؼаϰ�H�~����l
    if x_left + box_width > box_line[box_space - 1] or y_top > box_bottom: # or confidence < 0.9
      continue
	#�ھڰϰ�ƶq�A��W�x�sbox_top�Mbox��y_top
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
  #�B�z��l���|���D
  for i in range(box_space):
    #�P�_�ϰ줺�O�_����l
    if y_top_list[i].size != 0:
      diff_num = y_top_list[i][(y_top_list[i] - y_top_list[i].min()) > box_top_s] #�Ĥ@�h�@����
	  #�p���l�ƶq
	  #�p��y�{�G�N�Ĥ@�h����l�L�o���A�N�ѤU����l�ƶq���H�h���A�A�[�W�Ĥ@�h����l�ƶq
      if diff_num.size != 0:
        box_num[i] = (box_num[i] - diff_num.size) * 2 + diff_num.size
      else:
	  #box_top���b�P�@�h�ɡA�Nbox_top�ƶq���H�h���A�Y�@����ƶq
        if y_top_list[i].size != 0:
          if (y_top_box_list[i].max() - y_top_box_list[i].min()) > box_top_s:
            box_num[i] = box_num[i] * 2
  return box_num

box_num = box_count(detections, box_top_s, box_space, box_line, box_bottom)

print("Box_1:",box_num[0])
print("Box_2:",box_num[1])
print("Area_1:",str(int((box_num[0] / 15.0) * 100)) + "%")
print("Area_2:",str(int((box_num[1] / 14.0) * 100)) + "%")

#�N���G��ܦb�v����
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
  # �p�� Box �y��
  x_left = int(detection[2])
  y_top = int(detection[3])
  if x_left + box_width > box_line[box_space - 1] or y_top > box_bottom:
    continue
  # �b�v�����ХXBox��ɩM���O�B�H�߫�
  if label == "box":
    rectColor = (0, 255, 0)
    textCoord = (x_left, y_top - 10)
  elif label == "box_top":
    rectColor = (0, 255, 255)
    textCoord = (x_left, y_top + 20)
	
  cv2.rectangle(image, (x_left, y_top), (x_left + box_width, y_top + box_height), rectColor, 2)
  cv2.putText(image, pstring, textCoord, cv2.FONT_HERSHEY_DUPLEX, 1, rectColor, 2)

blue = (255, 0, 0)
#�ХX�U���O�ƶq
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
