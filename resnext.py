import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import layers\


class SplitBlock(layers.Layer):
    def __init__(self, hidden_dim=4, kernel_size=(3, 3)):
        super(SplitBlock, self).__init__()
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(input_shape[-1],
                                  kernel_size=self.kernel_size,
                                  padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=True):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


class ResNextBlock(layers.Layer):
    def __init__(self, hidden_dim=4, kernel_size=(3, 3),**kwargs):
        super(ResNextBlock, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.splits = []

    def build(self, input_shape):
        self.conv2a = layers.Conv2D(self.hidden_dim, (1, 1), padding='same')
        self.bn2a = layers.BatchNormalization()
        n_splits = input_shape[-1] // self.hidden_dim
        for _ in range(int(n_splits)):
            self.splits.append(SplitBlock(self.hidden_dim, self.kernel_size))
        self.conv2c = layers.Conv2D(input_shape[-1], (1, 1), padding='same')
        self.bn2c = layers.BatchNormalization()

    def call(self, input_tensor, training=True):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x_splits = [sb(x) for sb in self.splits]
        x = tf.concat(x_splits, axis=-1)
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return x