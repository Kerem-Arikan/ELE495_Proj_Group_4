import os
import time
from importlib import util
from argparse import ArgumentParser as argparser

GRAPH_DIR = '../frozen_graph'

argument_parser = argparser()
argument_parser.add_argument('--graphname', help="Name of the graph file.", default='detect.tflite')
argument_parser.add_argument('--webcamres', help="Requested resolution of the webcam.", default='1280x720')
argument_parser.add_argument('--score_threshold', help="Minimum score to confirm the detection as legitimate.", default=0.5)

arguments = argument_parser.parse_args()

GRAPH_PATH = os.path.join(GRAPH_DIR, arguments.graphname)

tflite_runtime_spec = util.find_spec('tflite_runtime')

print(GRAPH_PATH)
