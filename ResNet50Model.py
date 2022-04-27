import tensorflow as tf


class ResNet50Model(tf.keras.Model):
    def __init__(
            self,
            input_size,
            num_classes
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.base_model = tf.keras.applications.ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(None, None, 3)
        )
        self.flatten_1 = tf.keras.layers.Flatten()
        self.dropout_1 = tf.keras.layers.Dropout(rate=0.2)
        self.dense_1 = tf.keras.layers.Dense(self.num_classes, activation="softmax")

    def call(self, inputs):
        x = inputs
        x *= 255.0
        x = tf.keras.applications.resnet50.preprocess_input(x)
        x = self.flatten_1(x)
        x = self.dropout_1(x)
        output = self.dense_1(x)
        return output

    def get_config(self):
        return {
            "input_size": self.input_size,
            "num_classes": self.num_classes
        }

    @staticmethod
    def from_config(cls, config):
        cls(**config)
