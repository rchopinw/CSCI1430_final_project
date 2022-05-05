import tensorflow as tf
from ProcessARGS import GlobalArgs
from TensorFlowTools import ModelCheckPointSaver
from tensorflow_addons.optimizers import AdamW
import os


class ARGS:
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
                log_dir=GlobalArgs.log_dir + os.sep + "VanillaModel",
                update_freq="batch",
                profile_batch=0
            ),
            ModelCheckPointSaver(
                checkpoint_dir=GlobalArgs.model_dir + os.sep + "VanillaModel",
                model_id="VanillaModel",
                max_num_weights=5,
                save_weights=False
            )
        ]
    )

    ResNet50Model = dict(
        model_id="ResNet50Model",
        train_batch_size=32,
        validation_batch_size=32,
        train_epoch=10,
        train_buffer_size=256,
        validation_buffer_size=256,
        resize=(256, 256),
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
                log_dir=GlobalArgs.log_dir + os.sep + "ResNet50Model",
                update_freq="batch",
                profile_batch=0
            ),
            ModelCheckPointSaver(
                checkpoint_dir=GlobalArgs.model_dir + os.sep + "ResNet50Model",
                model_id="ResNet50Model",
                max_num_weights=5,
                save_weights=False
            )
        ]
    )

    XceptionModel = dict(
        model_id="XceptionModel",
        train_batch_size=32,
        validation_batch_size=32,
        train_epoch=10,
        train_buffer_size=256,
        validation_buffer_size=256,
        resize=(256, 256),
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
                log_dir=GlobalArgs.log_dir + os.sep + "XceptionModel",
                update_freq="batch",
                profile_batch=0
            ),
            ModelCheckPointSaver(
                checkpoint_dir=GlobalArgs.model_dir + os.sep + "XceptionModel",
                model_id="XceptionModel",
                max_num_weights=5,
                save_weights=False
            )
        ]
    )

    VGG19Model = dict(
        model_id="VGG19Model",
        train_batch_size=32,
        validation_batch_size=32,
        train_epoch=10,
        train_buffer_size=256,
        validation_buffer_size=256,
        resize=(350, 250),
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
                log_dir=GlobalArgs.log_dir + os.sep + "VGG19Model",
                update_freq="batch",
                profile_batch=0
            ),
            ModelCheckPointSaver(
                checkpoint_dir=GlobalArgs.model_dir + os.sep + "VGG19Model",
                model_id="VGG19Model",
                max_num_weights=5,
                save_weights=False
            )
        ]
    )

    InceptionResNetV2Model = dict(
        model_id="InceptionResNetV2Model",
        train_batch_size=32,
        validation_batch_size=32,
        train_epoch=10,
        train_buffer_size=256,
        validation_buffer_size=256,
        resize=(350, 250),
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
                log_dir=GlobalArgs.log_dir + os.sep + "InceptionResNetV2Model",
                update_freq="batch",
                profile_batch=0
            ),
            ModelCheckPointSaver(
                checkpoint_dir=GlobalArgs.model_dir + os.sep + "InceptionResNetV2Model",
                model_id="InceptionResNetV2Model",
                max_num_weights=5,
                save_weights=False
            )
        ]
    )

    InceptionV3Model = dict(
        model_id="InceptionV3Model",
        train_batch_size=32,
        validation_batch_size=32,
        train_epoch=20,
        train_buffer_size=256,
        validation_buffer_size=256,
        resize=(420, 300),
        auto_tune=tf.data.experimental.AUTOTUNE,
        train_optimizer=AdamW(
            learning_rate=1e-4,
            weight_decay=4e-4
        ),
        train_loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
            tf.keras.metrics.SparseCategoricalCrossentropy()
        ],
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=GlobalArgs.log_dir + os.sep + "InceptionV3Model",
                update_freq="batch",
                profile_batch=0
            ),
            ModelCheckPointSaver(
                checkpoint_dir=GlobalArgs.model_dir + os.sep + "InceptionV3Model",
                model_id="InceptionV3Model",
                max_num_weights=5,
                save_weights=False
            )
        ]
    )
