# Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense


class Classifier:

    def __init__(self, strategy):
        """
        Constructor which allows to select the desired strategy.
        """

        leaky_relu = tf.keras.layers.LeakyReLU()
        relu = tf.keras.layers.ReLU()

        model = tf.keras.models.Sequential()

        if strategy == 1:  # Strategy 1
            model.add(tf.keras.layers.Dense(6000, activation=leaky_relu, input_shape=(12210,)))
            model.add(tf.keras.layers.Dense(3000, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(1500, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(750, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(375, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(187, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(93, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(46, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(23, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(11, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(5, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(2, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        if strategy == 2:  # Strategy 2
            model.add(tf.keras.layers.Dense(1200, activation=leaky_relu, input_shape=(12210,)))
            model.add(tf.keras.layers.Dense(120, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(12, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        if strategy == 3:  # Strategy 3
            model.add(tf.keras.layers.Dense(1750, activation=leaky_relu, input_shape=(3500,)))
            model.add(tf.keras.layers.Dense(875, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(437, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(218, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(109, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(54, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(27, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(13, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(6, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(3, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        if strategy == 4:  # Strategy 4
            model.add(tf.keras.layers.Dense(350, activation=leaky_relu, input_shape=(3500,)))
            model.add(tf.keras.layers.Dense(35, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(3, activation=leaky_relu))
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        if strategy == 5:  # Strategy 5
            model.add(Dense(1750, activation=relu, input_shape=(12210,)))
            model.add(Dense(750, activation=relu))
            model.add(Dense(500, activation=relu))
            model.add(Dense(200, activation=relu))
            model.add(Dense(100, activation=relu))
            model.add(Dense(1, activation='sigmoid'))

        self.model = model
