import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers
from tensorflow.keras import Model  # Giorgio
from tensorflow.keras.layers import Layer   # Giorgio
from tensorflow.keras.layers import concatenate

tfd = tfp.distributions
tf.keras.backend.set_floatx('float64')


class deep_ProMPs_model_RGB(tf.keras.Model):

    def __init__(self):
        super(deep_ProMPs_model_RGB, self).__init__()

        self.MEAN_SIZE = 56
        self.PCA_SIZE = 63

        self.reg_l1_l2 = tf.keras.regularizers.l1_l2(float(0), float(0))
        self.NEURONS = [64]
        # Layers in "convolutional_block" Giorgio
        self.conv1 = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=self.reg_l1_l2)
        self.conv2 = layers.Conv2D(16, (3, 3), padding="same", kernel_regularizer=self.reg_l1_l2)
        self.conv3 = layers.Conv2D(8, (3, 3), padding="same", kernel_regularizer=self.reg_l1_l2)
        self.conv4 = layers.Conv2D(4, (3, 3), padding="same", kernel_regularizer=self.reg_l1_l2)
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2))
        self.drop1 = layers.Dropout(0.25)
        self.drop2 = layers.Dropout(0.25)
        self.drop3 = layers.Dropout(0.25)
        self.flat = layers.Flatten
        # Layers in "fully_connected_block" Giorgio
        self.output_mean = layers.Dense(self.MEAN_SIZE, activation="linear")
        self.output_pca = layers.Dense(self.PCA_SIZE, activation="linear")

    def call(self, input_layer):

        # Convolutional block
        x = self.conv1(input_layer)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.flat(name="feature_vec")(x)

        # 1 branch - mean prediction
        for dense_size in self.NEURONS:
            if dense_size == 64:
                x1 = layers.Dense(units=dense_size, activation="relu", kernel_regularizer=self.reg_l1_l2)(x)
            else:
                x1 = layers.Dense(units=dense_size, activation="relu", kernel_regularizer=self.reg_l1_l2)(x1)
        output_mean = self.output_mean(x1)

        # 2 branch - pca prediction
        for dense_size in self.NEURONS:
            if dense_size == 64:
                x2 = layers.Dense(units=dense_size, activation="relu", kernel_regularizer=self.reg_l1_l2)(x)
            else:
                x2 = layers.Dense(units=dense_size, activation="relu", kernel_regularizer=self.reg_l1_l2)(x2)
        output_pca = self.output_pca(x2)

        # concatenation of 1 branch and 2 branch
        output_layer = concatenate([output_mean, output_pca], axis=1)

        return output_layer

    def summary(self):
        """
        Define the model ad its summary
        """
        input_layer = layers.Input(shape=(32, 32, 3), name="encoded_image_input")   # econded
        model = tf.keras.Model(inputs=input_layer, outputs=self.call(input_layer))

        return model.summary()


if __name__ == "__main__":
    model = deep_ProMPs_model_RGB()
    model.summary()
