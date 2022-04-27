from TensorFlowTools import ModelCheckPointSaver
import os
import tensorflow as tf
import random


class ARGS:
    GlobalArgs = dict(
        random_seed=42,
        validation_split=0.2,
        num_classes=3,
        log_dir="/home/bangxi_xiao2/train_output/log",
        model_dir="/home/bangxi_xiao2/train_output/model"
    )

    TFRecordConfig = dict(
        directory='/home/bangxi_xiao2/data/images',
        save_directory='/home/bangxi_xiao2/data',
        size=5000,
        num_channels=3,
        image_size=(700, 500)
    )

    ResNet50TrainArgs = dict(
        model_id="ResNet50",
        train_batch_size=128,
        validation_batch_size=1024,
        train_epoch=50,
        train_buffer_size=2048,
        validation_buffer_size=1024,
        auto_tune=tf.data.experimental.AUTOTUNE,
        train_optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        train_loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalCrossentropy()
        ],
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=GlobalArgs["log_dir"] + os.sep + "ResNet50",
                update_freq="batch",
                profile_batch=0
            ),
            ModelCheckPointSaver(
                checkpoint_dir=GlobalArgs["model_dir"] + os.sep + "ResNet50",
                model_id="ResNet50",
                max_num_weights=5
            )
        ]
    )

    VGG16TrainArgs = dict(

    )

    InceptionTrainArgs = dict(

    )


random.seed(ARGS.GlobalArgs['random_seed'])
