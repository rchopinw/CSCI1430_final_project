import tensorflow as tf
import random


class ARGS:
    GlobalArgs = dict(
        random_seed=42,
        train_path='',
        validation_path='',
        test_path='',
        validation_split=0.2,
    )

    TFRecordConfig = dict(
        directory='D:/Brown Learning Materials/CSCI1430/final/archive/images',
        save_directory='D:/Brown Learning Materials/CSCI1430/final/archiv',
        size=5000,
        image_size=(700, 500)
    )

    ImagePreprocessArgs = dict(
        rotation_range=30,
        width_shift_range=10,
        height_shift_range=10,
        brightness_range=(0.2, 0.8),
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )

    ImageDataGeneratorArgs = dict(
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        target_size=(512, 512)
    )

    ResNet50TrainArgs = dict(
        train_batch_size=128,
        train_epoch=50,
        train_buffer_size=1024,
        validation_buffer_size=1024,
        auto_tune=tf.data.experimental.AUTOTUNE,
        train_optimizer=tf.keras.optimizers.Adam(),
        train_loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[],
        callbacks=[]
    )

    VGG16TrainArgs = dict(

    )

    InceptionTrainArgs = dict(

    )


random.seed(ARGS.GlobalArgs['random_seed'])