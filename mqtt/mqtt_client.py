import paho.mqtt.client as mqtt
import cv2
import numpy as np
import time

'''
def on_connect(client, userdata, flags, rc):
    print("链接")
    print("Connected with result code: " + str(rc))
'''

net = cv2.dnn_DetectionModel('cfg/enet-coco.cfg', 'model/enetb0-coco_final.weights')
#net = cv2.dnn_DetectionModel('cfg/yolov4-tiny.cfg', 'model/yolov4-tiny.weights')
net.setInputSize(416, 416)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)

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

def on_message(client, userdata, msg):
    #print("消息内容")
    grid_43 = "0"
    grid_44 = "0"
    grid_45 = "0"
    grid_46 = "0"
    grid_47 = "0"
    grid_48 = "0"
    grid_12 = "0"
    grid_13 = "0"
    grid_14 = "0"
    print(msg.topic + " " + str(msg.payload))
    if msg.payload.decode()== 'take_picture':
        cap = cv2.VideoCapture(0)
        time.sleep(0.1)
        ret, frame = cap.read()
        cap.release()
        
        classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.5)
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
                '''
                pstring = str(int(100 * confidence)) + "%" #信心度
                x_left, y_top, width, height = box
                boundingBox = [
                    (x_left, y_top), #左上頂點
                    (x_left, y_top + height), #左下頂點
                    (x_left + width, y_top + height), #右下頂點
                    (x_left + width, y_top) #右上頂點
                ]
                rectColor = (0, 0, 255)
                textCoord = (x_left, y_top - 10) #文字位置
                # 在影像中標出Box邊界和類別、信心度
                cv2.rectangle(frame, boundingBox[0], boundingBox[2], rectColor, 2)
                cv2.putText(frame, pstring, textCoord, cv2.FONT_HERSHEY_DUPLEX, 1, rectColor, 2)
                '''
        result = grid_43 + " " + grid_44 + " " + grid_45 + " " + grid_46 + " " + grid_47 + " " + grid_48 + " " + grid_12 + " " + grid_13 + " " + grid_14
        print(result)
        
        #cv2.imwrite("test.jpg", frame)
        #time.sleep(3)
        
        client.publish("callback_N1",result,0)
        print("Waiting 1 seconds")
        #time.sleep(1)
        #client.publish()


'''
#   订阅回调
def on_subscribe(client, userdata, mid, granted_qos):
    print("订阅")
    print("On Subscribed: qos = %d" % granted_qos)
    pass
 
#   取消订阅回调
def on_unsubscribe(client, userdata, mid, granted_qos):
    print("取消订阅")
    print("On unSubscribed: qos = %d" % granted_qos)
    pass
 
 
#   发布消息回调
def on_publish(client, userdata, mid):
    print("发布消息")
    print("On onPublish: qos = %d" % mid)
    pass

'''


HOST = "192.168.60.103"#"172.20.10.12"
PORT = 1883

client = mqtt.Client()
#client.on_connect = on_connect
client.on_message = on_message

client.connect(HOST, PORT, 600)
client.subscribe('take_picture', qos=0)
client.loop_forever()


'''
while True:
    
    client.publish(topic='mqtt11', payload='hello MQTT', qos=0, retain=False)
    time.sleep(2)
'''    
    
#client.on_publish = on_publish
#client.on_disconnect = on_disconnect
#client.on_unsubscribe = on_unsubscribe
#client.on_subscribe = on_subscribe



