import paho.mqtt.client as mqtt
import cv2
import numpy as np
import time

#def on_connect(client, userdata, flags, rc):
    #print("Connected with result code: " + str(rc))
to_server_count = 12
person_in_time = 600
in_timer = 0
loop_timer = 0
avg_count = 0
N1_result = ""
N2_result = ""
N3_result = ""
N4_result = ""
All_result = "00000000000000000000000000000000"

def string_or(s1,s2):    
    return ''.join(chr(ord(a) ^ ord(b)) for a,b in zip(s1,s2))

def on_message(client, userdata, msg):
    global in_timer, avg_count, N1_result, N2_result, N3_result, N4_result
    
    #print(msg.topic + "," + str(msg.payload))
    if msg.payload.decode() == 'key_in':
        #print(msg.topic + "," + msg.payload.decode())
        in_timer = time.time()
        avg_count = 0
    elif msg.topic == 'callback_N1':
        N1_result = msg.payload.decode()
    elif msg.topic == 'callback_N2':
        N2_result = msg.payload.decode()
    elif msg.topic == 'callback_N3':
        N3_result = msg.payload.decode()
    elif msg.topic == 'callback_N4':
        N4_result = msg.payload.decode()
    
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
    #print("发布消息")
    #print("On onPublish: qos = %d" % mid)
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
#client.loop_forever()

while True:
    if in_timer != 0:
        client.publish(topic='take_picture', payload='take_picture', qos=0, retain=False)
        time.sleep(5)
        if N1_result != "" and N2_result != "" and N3_result != "" and N4_result != "":
            avg_count = avg_count + 1
            N1_split = N1_result.split(" ")
            N2_split = N2_result.split(" ")
            N3_split = N3_result.split(" ")
            N4_split = N4_result.split(" ")
            All_result_old = All_result
            All_result = N1_split[0] + N1_split[1] + N1_split[2] + N1_split[3] + N2_result[0] + N2_result[1] + N2_result[2] + N3_split[0] + N3_split[1] + N3_split[2] + N3_split[3] + N4_split[0] + N4_split[1] + N4_split[2] + N1_split[4] + N1_split[5] + N1_split[6] + N1_split[7] + N1_result[8] + N2_split[3] + N2_split[4] + N2_split[5] + N2_split[6] + N2_result[7] +  N3_split[4] + N3_split[5] + N3_split[6] + N3_split[7] + N4_split[3] + N4_split[4] + N4_split[5] + N4_split[6]
            All_result = string_or(All_result_old, All_result)
            if avg_count == to_server_count:
                client.publish(topic='grid_status', payload=All_result, qos=0, retain=False)
        loop_timer = time.time()
        if loop_timer - in_timer < person_in_time:
            N1_result = ""
            N2_result = ""
            N3_result = ""
            N4_result = ""
        else:
            All_result = "00000000000000000000000000000000"
            in_timer = 0




