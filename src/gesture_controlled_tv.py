import os
import time
from importlib import util
from argparse import ArgumentParser as argparser
import cv2 as cv
from threading import Thread
import numpy as np

GRAPH_DIR = '../frozen_graph'

argument_parser = argparser()
argument_parser.add_argument('--graphname', help="Name of the graph file.", default='detect.tflite')
argument_parser.add_argument('--webcamres', help="Requested resolution of the webcam.", default='1280x720')
argument_parser.add_argument('--labelmap', help="Name of the label map", default='labelmap.txt')
argument_parser.add_argument('--keymap', help="Name of the ri key map", default='keymap.txt')
argument_parser.add_argument('--score_threshold', help="Minimum score to confirm the detection as legitimate.", default=0.5)
argument_parser.add_argument('--use_delegates', help="Selecting if delegates will be used from the interpreter.", default=False)

arguments = argument_parser.parse_args()

GRAPH_PATH = os.path.join(GRAPH_DIR, arguments.graphname)
SCORE_THRESHOLD = float(arguments.score_threshold)

LABELMAP_PATH = os.path.join(GRAPH_DIR, arguments.labelmap)
KEYMAP_PATH = os.path.join(os.getcwd(), arguments.keymap)

labelmap_file = open(LABELMAP_PATH, 'r')
label_list = [currline.strip() for currline in labelmap_file.readlines()]
labelmap_file.close()

resolution_width, resolution_height = arguments.webcamres.split('x')
resolution_width = int(resolution_width)
resolution_height = int(resolution_height)

print(arguments.use_delegates)

tflite_exists = util.find_spec('tflite_runtime')
if tflite_exists:
    from tflite_runtime.interpreter import Interpreter
    if arguments.use_delegates: 
        from tflite_runtime.interpreter import load_delegate
    
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if arguments.use_delegates: 
        from tensorflow.lite.python.interpreter import load_delegate

if arguments.use_delegates:
    delegate_list = [load_delegate('libedgetpu.so.1.0')] 
    interp = Interpreter(model_path=GRAPH_PATH, experimental_delegates=delegate_list)
else:
    interp = Interpreter(model_path=GRAPH_PATH)

interp.allocate_tensors()

input_info = interp.get_input_details()
output_info = interp.get_output_details()

input_avg = input_standart = 127.5

framerate = 1
cv.getTickFrequency()


capture = cv.VideoCapture(0)
#capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
#capture.set(3, resolution_width)
#capture.set(4, resolution_height)

(ret, curr_frame) = capture.read()

#camera_read_thread = Thread(target=capture.read, args=())
#camera_read_thread.start()

time.sleep(1)
while True:
    tick_count = cv.getTickCount()
    
    (ret, captured_frame) = capture.read()

    #captured_frame = cv.cvtColor(captured_frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(captured_frame, (input_info[0]['shape'][1], input_info[0]['shape'][2]))

    input_dat = np.expand_dims(frame, axis=0)
    
    if input_info[0]['dtype'] == np.float32:
        input_dat = (np.float32(input_dat) - input_avg) / input_standart

    interp.set_tensor(input_info[0]['index'], input_dat)
    interp.invoke()

    box_list = interp.get_tensor(output_info[0]['index'])[0]
    class_list = interp.get_tensor(output_info[1]['index'])[0]
    score_list = interp.get_tensor(output_info[2]['index'])[0]

    score_list = list(score_list)
    max_idx = score_list.index(max(score_list))
    if(max(score_list) > SCORE_THRESHOLD):
        print(label_list[int(class_list[max_idx])], " with score of ", max(score_list))
    
    if cv.waitKey(1) == ord('q'):
        break

capture.release()