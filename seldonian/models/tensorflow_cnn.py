# tensorflow_cnn.py

from seldonian.models.tensorflow_model import SupervisedTensorFlowBaseModel

import tensorflow as tf


class TensorFlowCNN(SupervisedTensorFlowBaseModel):
    def __init__(self, **kwargs):
        """Base class for Supervised learning Seldonian
        models implemented in TensorFlow

        """
        super().__init__()

    def create_model(self, **kwargs):
        """Create the TensorFlow model and return it"""
        num_classes = 10
        input_shape = (28, 28, 1)
        cnn = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32,
                    (5, 5),
                    padding="same",
                    activation="relu",
                    input_shape=input_shape,
                ),
                tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                tf.keras.layers.MaxPool2D(strides=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        cnn.build(input_shape=input_shape)
        return cnn
