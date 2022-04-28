import tensorflow as tf


def get_vanilla_model(
        input_size,
        num_classes,
        resize,
        translation,
        zoom,
        contrast,
        flip,
        model_id="VanillaModel"
):
    input_layer = tf.keras.Input(
        shape=input_size,
        name=model_id+"_input_layer"
    )
    aug_resize = tf.keras.layers.Resizing(*resize)(input_layer)
    aug_translation = tf.keras.layers.RandomTranslation(*translation)(aug_resize)
    aug_zoom = tf.keras.layers.RandomZoom(*zoom)(aug_translation)
    aug_contrast = tf.keras.layers.RandomContrast(contrast)(aug_zoom)
    aug_flip = tf.keras.layers.RandomFlip(flip)(aug_contrast)
    # block 1
    x = tf.keras.layers.Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv1")(aug_flip)
    x = tf.keras.layers.Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv2")(x)
    x = tf.keras.layers.MaxPool2D(2, name="block1_pool")(x)

    # block 2
    x = tf.keras.layers.Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv1")(x)
    x = tf.keras.layers.Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv2")(x)
    x = tf.keras.layers.MaxPool2D(2, name="block2_pool")(x)

    # block 3
    x = tf.keras.layers.Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv1")(x)
    x = tf.keras.layers.MaxPool2D(2, name="block3_pool_1")(x)
    x = tf.keras.layers.Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv2")(x)
    x = tf.keras.layers.MaxPool2D(2, name="block3_pool_2")(x)
    x = tf.keras.layers.Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv3")(x)
    x = tf.keras.layers.MaxPool2D(2, name="block3_pool_3")(x)

    # flatten
    x = tf.keras.layers.Flatten(name="flatten")(x)

    # Dense + Dropout
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_2")(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_3")(x)

    vanilla_model = tf.keras.Model(inputs=input_layer, outputs=output)

    return vanilla_model


def get_res_net_50_model(
        input_size,
        num_classes,
        resize,
        depth=3,
        model_id="ResNet50"
):
    base_model = tf.keras.applications.ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(None, None, depth)
    )
    input_layer = tf.keras.Input(shape=input_size, name=model_id+"_input_layer")
    x = tf.keras.layers.Rescaling(255.0)(input_layer)
    x = tf.keras.preprocessing.image.smart_resize(x, resize)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dropout(rate=0.1)(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    m = tf.keras.Model(
        inputs=input_layer,
        outputs=x
    )
    return m


def get_xception_model(
        input_size,
        num_classes,
        resize,
        depth=3,
        model_id="Xception"
):
    base_model = tf.keras.applications.Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(None, None, depth)
    )
    input_layer = tf.keras.Input(shape=input_size, name=model_id+"_input_layer")
    x = tf.keras.layers.Rescaling(255.0)(input_layer)
    x = tf.keras.preprocessing.image.smart_resize(x, resize)
    x = tf.keras.applications.xception.preprocess_input(x)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dropout(rate=0.1)(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    m = tf.keras.Model(
        inputs=input_layer,
        outputs=x
    )
    return m
