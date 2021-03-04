from ctypes import *
import cv2
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

lib = CDLL("yolo_cpp_dll.dll", RTLD_GLOBAL)
load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

load_meta = lib.get_metadata
load_meta.argtypes = [c_char_p]
load_meta.restype = METADATA

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float,
 c_float,POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

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

#----讀取模型----
configPath = "cfg/custom-yolov4-detector.cfg"
weightPath = "backup/custom-yolov4-detector_last.weights"
metaPath = "data/obj.data"

network = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
metadata = load_meta(metaPath.encode("ascii"))
class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
#----讀取模型結束----

#讀取影像
imagePath = "box_image/151340.jpg"
#im = load_image(imagePath.encode("ascii"), 0, 0)
im = cv2.imread(imagePath)
im, im_arr = array_to_image(im)

#預測影像
detections = detect_image(network, class_names, im)
#[label, confidence, [x_center, y_center, width, height]]

'''
#顯示預測影像
image = cv2.imread(imagePath)
print(str(len(detections))+" Results")
for detection in detections:
  label = detection[0]
  confidence = detection[1]
  pstring = label + ": " + str(int(100 * confidence)) + "%"
  bounds = detection[2]
  shape = image.shape
  box_height = int(bounds[3])
  box_width = int(bounds[2])
  # 計算 Box 座標
  x_left = int(bounds[0] - bounds[2]/2)
  y_top = int(bounds[1] - bounds[3]/2)
  boundingBox = [
      (x_left, y_top), #左上頂點
      (x_left, y_top + box_height), #左下頂點
      (x_left + box_width, y_top + box_height), #右下頂點
      (x_left + box_width, y_top) #右上頂點
  ]
  # 在影像中標出Box邊界和類別、信心度
  rectColor = (0, 255, 0)
  textCoord = (x_left, y_top + 20)
  cv2.rectangle(image, boundingBox[0], boundingBox[2], rectColor, 2)
  cv2.putText(image, pstring, textCoord, cv2.FONT_HERSHEY_DUPLEX, 1, rectColor, 2)

cv2.imwrite("predict.jpg", image)
cv2.namedWindow("box_count", cv2.WINDOW_NORMAL)
cv2.imshow("box_count", image)
cv2.waitKey(0)
'''
