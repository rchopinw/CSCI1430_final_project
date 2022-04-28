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
        size=1000,
        num_channels=3,
        image_size=(700, 500)
    )

    VanillaModel = dict(
        model_id="VanillaModel",
        train_batch_size=32,
        validation_batch_size=32,
        train_epoch=50,
        train_buffer_size=256,
        validation_buffer_size=256,
        auto_tune=tf.data.experimental.AUTOTUNE,
        resize=(210, 150),
        translation=(0.1, 0.1),
        zoom=(0.1, 0.1),
        contrast=0.3,
        flip="horizontal",
        train_optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),
        train_loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            'sparse_categorical_accuracy'
        ],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=3),
            tf.keras.callbacks.TensorBoard(
                log_dir=GlobalArgs["log_dir"] + os.sep + "VanillaModel",
                update_freq="batch",
                profile_batch=0
            ),
            ModelCheckPointSaver(
                checkpoint_dir=GlobalArgs["model_dir"] + os.sep + "VanillaModel",
                model_id="VanillaModel",
                max_num_weights=5,
                save_weights=False
            )
        ]
    )

    ResNet50Model = dict(
        model_id="ResNet50",
        train_batch_size=32,
        validation_batch_size=32,
        train_epoch=10,
        train_buffer_size=256,
        validation_buffer_size=256,
        auto_tune=tf.data.experimental.AUTOTUNE,
        train_optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        train_loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
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
                max_num_weights=5,
                save_weights=False
            )
        ]
    )

    XceptionModel = dict(
        model_id="Xception",
        train_batch_size=32,
        validation_batch_size=32,
        train_epoch=10,
        train_buffer_size=256,
        validation_buffer_size=256,
        auto_tune=tf.data.experimental.AUTOTUNE,
        train_optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        train_loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
            tf.keras.metrics.SparseCategoricalCrossentropy()
        ],
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=GlobalArgs["log_dir"] + os.sep + "Xception",
                update_freq="batch",
                profile_batch=0
            ),
            ModelCheckPointSaver(
                checkpoint_dir=GlobalArgs["model_dir"] + os.sep + "Xception",
                model_id="Xception",
                max_num_weights=5,
                save_weights=False
            )
        ]
    )

    VGG16TrainArgs = dict(

    )

    InceptionTrainArgs = dict(

    )


random.seed(ARGS.GlobalArgs['random_seed'])
