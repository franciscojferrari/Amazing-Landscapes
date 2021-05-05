import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_addons as tfa

from typing import Any


class Discriminator(tfkl.Layer):
    def __init__(
            self,
            name="discriminator",
            **kwargs,
    ):
        super(Discriminator, self).__init__(name=name, **kwargs)
        """PatchGAN discriminator"""
        self.leakyrelu = tfkl.LeakyReLU(alpha=0.2)
        self.layer1 = tfkl.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            padding="same",
            activation=tfkl.LeakyReLU(alpha=0.2),
        )

        self.layer2 = tfkl.Conv2D(
            filters=128,
            kernel_size=4,
            strides=2,
            padding="same",
        )
        self.norm2 = tfa.layers.InstanceNormalization(groups=128)

        self.layer3 = tfkl.Conv2D(
            filters=256,
            kernel_size=4,
            strides=2,
            padding="same",
        )
        self.norm3 = tfa.layers.InstanceNormalization(groups=256)

        self.layer4 = tfkl.Conv2D(
            filters=512,
            kernel_size=4,
            strides=2,
            padding="same",
        )
        self.norm4 = tfa.layers.InstanceNormalization(groups=512)

        self.output_layer = tfkl.Conv2D(
            filters=1,
            kernel_size=4,
            strides=1,
            padding="same")

    def call(self, image, mask):

        inputs = tf.concat([image, mask], axis=-1)

        x = self.layer1(inputs)
        x = self.leakyrelu(self.norm2(self.layer2(x)))
        x = self.leakyrelu(self.norm3(self.layer3(x)))
        x = self.leakyrelu(self.norm4(self.layer4(x)))

        out = self.output_layer(x)

        return out
