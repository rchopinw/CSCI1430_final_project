import tensorflow as tf
import random


class ARGS:
    GlobalArgs = dict(
        random_seed=42,
        train_path='',
        validation_path='',
        test_path='',
        rescale=1 / 255,
        tf_record_size=5000,
        img_size=(700, 500),
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
        buffer_size=512,
        auto_tune=tf.data.experimental.AUTOTUNE,
        train_optimizer=tf.keras.optimizers.Adam(),
        train_loss=tf.keras.losses.SparseCategoricalCrossentropy()
    )

    VGG16TrainArgs = dict(

    )

    InceptionTrainArgs = dict(

    )


random.seed(ARGS.GlobalArgs['random_seed'])