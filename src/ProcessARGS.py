import tensorflow as tf
import random


class GlobalArgs:
    random_seed = 42
    validation_split = 0.2
    num_classes = 3
    log_dir = "../train_output/log"
    model_dir = "../train_output/model"
    face_recognition_dnn_dir = '../face_recognition_model/res10_300x300_ssd_iter_140000.caffemodel'
    face_recognition_dnn_prototxt_dir = '../face_recognition_model/deploy.prototxt.txt'
    # log_dir = "D:/Brown Learning Materials/CSCI1430/train_logs/log"
    # model_dir = "D:/Brown Learning Materials/CSCI1430/train_logs/model"
    # face_recognition_dnn_dir = \
    #     'D:/Brown Learning Materials/CSCI1430/face_recognition_models/res10_300x300_ssd_iter_140000.caffemodel'
    # face_recognition_dnn_prototxt_dir = \
    #     'D:/Brown Learning Materials/CSCI1430/face_recognition_models/deploy.prototxt.txt'


class TFRecordConfigArgs:
    train_directory = '../data/train_images'  # original images are no longer saved
    train_save_directory = '../data/train_validation_data'
    train_size = 5000
    test_directory = '../data/test_images'  # original images are no longer saved
    test_save_directory = '../data/test_data'
    # test_directory = "D:/Brown Learning Materials/CSCI1430/archive/images"
    # test_save_directory = "D:/Brown Learning Materials/CSCI1430/archive/test_tfrecords"
    test_size = 1000
    test_batch_size = 32
    test_buffer_size = 256
    num_channels = 3
    image_size = (700, 500)
    auto_tune = tf.data.experimental.AUTOTUNE


random.seed(GlobalArgs.random_seed)
