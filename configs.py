import tensorflow as tf
from easydict import EasyDict as edict

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

AUTOTUNE = tf.data.experimental.AUTOTUNE
PATH = '/home/cheon/workspace/data/gauge/'
TRAIN_PATH = PATH + 'resize/'
TEST_PATH = PATH + 'resize/test/'
TRAIN_LABEL = '/home/cheon/workspace/data/gauge/gauge_labels.json'
TEST_LABEL = '/home/cheon/workspace/data/gauge/gauge_labels.json'

NUM_KEYPOINTS = 4
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
HEATMAP_SIZE = 56 # IMG_SIZE / 4
HEATMAP_SHAPE = (NUM_KEYPOINTS, HEATMAP_SIZE, HEATMAP_SIZE)
LABEL_SHAPE = (4, 2)
POINTS_LABELS = ['min', 'max', 'center', 'pointer']

POSE_RESNET = edict()
POSE_RESNET.NUM_LAYERS = 50
POSE_RESNET.DECONV_WITH_BIAS = True
POSE_RESNET.NUM_DECONV_LAYERS = 3
POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256] # original = [256, 256, 256]
POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
POSE_RESNET.FINAL_CONV_KERNEL = 1
POSE_RESNET.TARGET_TYPE = 'gaussian'
POSE_RESNET.HEATMAP_SIZE = [HEATMAP_SIZE, HEATMAP_SIZE]  # width * height, ex: 24 * 32
POSE_RESNET.SIGMA = 2

VAL_RATIO = 0.2
BATCH_SIZE = 16
TEST_BATCH_SIZE = 1
BN_MOMENTUM = 0.9 # tf = 1 - BN_MOMENTUM(pytorch)
LEARNING_RATE = 1e-5
NUM_EPOCH = 250
