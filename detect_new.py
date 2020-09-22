from ctypes import *
import cv2
import matplotlib.pyplot as plt
import numpy as np

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

lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
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
    predictions = remove_negatives(detections, class_names, num)
	free_detections(detections, num)

    return sorted(predictions, key=lambda x: -x[1])

configPath = "./cfg/custom-yolov4-detector.cfg"
weightPath = "custom-yolov4-detector_last.weights"
metaPath = "./data/obj.data"

network = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
metadata = load_meta(metaPath.encode("ascii"))
class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]

imagePath = "dataset/test/3.jpg"
im = load_image(imagePath.encode("ascii"), 0, 0)
#�w��
detections = detect_image(network, class_names, im)

image = cv2.imread(imagePath)
print(str(len(detections))+" Results")
box_num = 0
box_top_num = 0
for detection in detections:
  label = detection[0]
  confidence = detection[1]
  pstring = label + ": " + str(int(100 * confidence)) + "%"
  bounds = detection[2]
  shape = image.shape
  box_height = int(bounds[3])
  box_width = int(bounds[2])
  # �p�� Box �y��
  x_left = int(bounds[0] - bounds[2]/2)
  y_top = int(bounds[1] - bounds[3]/2)
  boundingBox = [
      (x_left, y_top), #���W���I
      (x_left, y_top + box_height), #���U���I
      (x_left + box_width, y_top + box_height), #�k�U���I
      (x_left + box_width, y_top) #�k�W���I
  ]
  # �b�v�����ХXBox��ɩM���O�B�H�߫�
  if label == "box":
    rectColor = (0, 255, 0)
    textCoord = (x_left, y_top - 10)
    box_num = box_num + 1
  elif label == "box_top":
    rectColor = (0, 255, 255)
    textCoord = (x_left, y_top + 20)
    box_top_num = box_top_num + 1
	
  cv2.rectangle(image, boundingBox[0], boundingBox[2], rectColor, 2)
  cv2.putText(image, pstring, textCoord, cv2.FONT_HERSHEY_DUPLEX, 1, rectColor, 2)

blue = (255, 0, 0)
#�ХX�U���O�ƶq
cv2.putText(image, "Box: " + str(box_num), (10,80), cv2.FONT_HERSHEY_DUPLEX, 2, blue, 2)
cv2.putText(image, "Box_top: " + str(box_top_num), (10,160), cv2.FONT_HERSHEY_DUPLEX, 2, blue, 2)
cv2.imwrite("test01.jpg", image)

