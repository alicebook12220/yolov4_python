import paho.mqtt.client as mqtt
import cv2
import numpy as np
import time

#def on_connect(client, userdata, flags, rc):
    #print("Connected with result code: " + str(rc))

in_timer = 0
loop_timer = 0
cam1_result = ""
def on_message(client, userdata, msg):
    global in_timer, loop_timer
    
    #print(msg.topic + "," + str(msg.payload))
    if msg.payload.decode() == 'N1_OK':
        print(msg.topic + "," + msg.payload.decode())
        in_timer = time.time()
    elif msg.topic == 'callback_N1':
        cam1_result = msg.payload.decode()
        loop_timer = time.time()
        #print()
    #print(msg.payload.decode())
    
#   订阅回调
#def on_subscribe(client, userdata, mid, granted_qos):
    #print("On Subscribed: qos = %d" % granted_qos)
    #pass

#   取消订阅回调
#def on_unsubscribe(client, userdata, mid):
    #print("取消订阅")
    #print("On unSubscribed: qos = %d" % mid)
    #pass
 

#   发布消息回调

def on_publish(client, userdata, mid):
    print("发布消息")
    print("On onPublish: qos = %d" % mid)
    pass


#   断开链接回调
#def on_disconnect( client, userdata, rc):
    #print("断开链接")
    #print("Unexpected disconnection rc = " + str(rc))
    #pass


client = mqtt.Client()
#client.on_connect = on_connect
client.on_message = on_message
client.on_publish = on_publish
#client.on_disconnect = on_disconnect
#client.on_unsubscribe = on_unsubscribe
#client.on_subscribe = on_subscribe


HOST = "192.168.60.103"#"172.20.10.12"
PORT = 1883
client.connect(HOST, PORT, 600) # 600为keepalive的时间间隔

client.subscribe('callback_N1',qos=0)  #N1 rasberry pi
client.subscribe('callback_N2',qos=0)  #N2 rasberry pi
client.subscribe('callback_N3',qos=0)  #N3 rasberry pi
client.subscribe('callback_N4',qos=0)  #N4 rasberry pi

client.loop_start()  ## open another workflow

while True:
    
    client.publish(topic='mqtt_test',payload='take_picture',qos=0,retain=False)
    
    time.sleep(5)
    print(in_timer)
#client.loop_forever()




