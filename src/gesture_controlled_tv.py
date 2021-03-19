import os
import subprocess
import argparse
import cv2
import numpy as np
import sys
import time
import importlib.util
from video_capture import video_capture



def label2key(keys_path, gestures_path):
    dic_data={}
    try:
        keys = open(keys_path, "r")
        gestures = open(gestures_path,"r")
        while True:
            dumbkey = keys.readline()
            dumbgesture = gestures.readline()
            # End of file check
            if dumbkey == '':
                break
            if dumbgesture == '':
                break
            # Check if there is new line symbol.
            if dumbkey[-1]=="\n":
                dumbkey = dumbkey[:-1]
            if dumbgesture[-1]=='\n':
                dumbgesture = dumbgesture[:-1]

            dic_data[dumbgesture]=dumbkey
        keys.close()
        gestures.close()
    except IOError:
        print("IOError")
        print("gestures_path="+gestures_path)
        print("keys_path="+keys_path)
    print(dic_data)
    return dic_data


parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default='frozen_graph')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--allow_cam_display', help='Display the captured image or not.', default=True)

parser.add_argument('--tvname', help="Name of the tv name.", default="kerem-tv") # kerem-tv Samsung_TV starbox

parser.add_argument('--response_rate', help="How quick should the detection be?", default=5)

parser.add_argument('--count2send_lim', help="back to back numbers", default=4)

args = parser.parse_args()

send_lim = int(args.count2send_lim)

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

label_key_map = label2key(keys_path="/home/pi/ELE495_Proj_Group_4/src/keymap.txt", gestures_path="/home/pi/ELE495_Proj_Group_4/frozen_graph/labelmap.txt")

counter_limit = 2

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

CWD_PATH = "/home/pi/ELE495_Proj_Group_4" #os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

frame_rate_calc = 1
freq = cv2.getTickFrequency()

videostream = video_capture(resolution=(imW,imH),framerate=30).start()
time.sleep(1)
queue=[]
count2send=0
prev_obj_name = 0
counter = 0
while True:

    t1 = cv2.getTickCount()
    
    frame1 = videostream.read()

    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0] 
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    #num = interpreter.get_tensor(output_details[3]['index'])[0] 

    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            '''
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            '''
            object_name = labels[int(classes[i])] 
            '''
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) 
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
            label_ymin = max(ymin, labelSize[1] + 10) 
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            '''
            key_name = label_key_map[object_name]
            command_string = "irsend SEND_ONCE" + " " + args.tvname + " " + key_name
            if(prev_obj_name == object_name):
                counter += 1
                if(counter==counter_limit):
                    print(command_string,"only detected not sent")
                    if(object_name in ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']):
                        queue.append(command_string)
                    else:
                        os.system(command_string)
                    counter = 0
            else:
                counter = 0
            prev_obj_name = object_name
    if(len(queue)>=2) or count2send>=send_lim:
        print("sending", len(queue),"command(s)")
        print("Queue: ", queue)
        for s in queue:
            os.system(s)
            time.sleep(0.5)
            print(s,"SENDING")
        queue=[]
        count2send=0
        
    if(len(queue)>0):
                count2send+=1;print("line coun2send is:",count2send)
    '''
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    cv2.imshow('Object detector', frame)
    '''
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1 #;print(frame_rate_calc)
    
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videostream.stop()

