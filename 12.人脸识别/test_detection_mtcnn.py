import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import sys
import copy
import detect_face
import nn4 as network
import random

import sklearn

from sklearn.externals import joblib

# face detection parameters
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor

# facenet embedding parameters

model_dir = './model_check_point/model.ckpt-500000'  # "Directory containing the graph definition and checkpoint files.")
image_size = 96  # "Image size (height, width) in pixels."
pool_type = 'MAX'  # "The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn = False  # "Enables Local Response Normalization after the first layers of the inception network."
seed = 42,  # "Random seed."
batch_size = None  # "Number of images to process in a batch."


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


if __name__ == '__main__':
    # restore mtcnn model
    print('Creating networks and loading parameters')
    gpu_memory_fraction = 1.0
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')

    image = cv2.imread(sys.argv[1])

    find_results = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if gray.ndim == 2:
        img = to_rgb(gray)

    bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    nrof_faces = bounding_boxes.shape[0]  # number of faces

    num = -1
    for face_position in bounding_boxes:
        num += 1
        face_position = face_position.astype(int)

        # draw face
        cv2.rectangle(image, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 5)

        # draw feature points
        cv2.circle(image, (points[0][num], points[5][num]), 5, (0, 255, 0), -1)
        cv2.circle(image, (points[1][num], points[6][num]), 5, (0, 255, 0), -1)
        cv2.circle(image, (points[2][num], points[7][num]), 5, (0, 255, 0), -1)
        cv2.circle(image, (points[3][num], points[8][num]), 5, (0, 255, 0), -1)
        cv2.circle(image, (points[4][num], points[9][num]), 5, (0, 255, 0), -1)

    # show result
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Show Result", image)
    cv2.waitKey(0)
