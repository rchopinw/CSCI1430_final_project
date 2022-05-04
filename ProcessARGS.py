import tensorflow as tf
import random


class GlobalArgs:
    random_seed = 42
    validation_split = 0.2
    num_classes = 3
    log_dir = "/home/bangxi_xiao2/train_output/log"
    model_dir = "/home/bangxi_xiao2/train_output/model"
    face_recognition_dnn_dir = '/home/bangxi_xiao2/res10_300x300_ssd_iter_140000.caffemodel'
    face_recognition_dnn_prototxt_dir = '/home/bangxi_xiao2/deploy.prototxt.txt'
    # log_dir = "D:/Brown Learning Materials/CSCI1430/train_logs/log"
    # model_dir = "D:/Brown Learning Materials/CSCI1430/train_logs/model"
    # face_recognition_dnn_dir = \
    #     'D:/Brown Learning Materials/CSCI1430/face_recognition_models/res10_300x300_ssd_iter_140000.caffemodel'
    # face_recognition_dnn_prototxt_dir = \
    #     'D:/Brown Learning Materials/CSCI1430/face_recognition_models/deploy.prototxt.txt'


class TFRecordConfigArgs:
    train_directory = '/home/bangxi_xiao2/data/images'
    train_save_directory = '/home/bangxi_xiao2/data'
    train_size = 5000
    test_directory = '/home/bangxi_xiao2/data/images'
    test_save_directory = '/home/bangxi_xiao2/data/test_tfrecords'
    # test_directory = "D:/Brown Learning Materials/CSCI1430/archive/images"
    # test_save_directory = "D:/Brown Learning Materials/CSCI1430/archive/test_tfrecords"
    test_size = 1000
    test_batch_size = 32
    test_buffer_size = 256
    num_channels = 3
    image_size = (700, 500)
    auto_tune = tf.data.experimental.AUTOTUNE


random.seed(GlobalArgs.random_seed)
