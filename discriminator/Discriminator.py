import tensorflow as tf
import tensorflow.keras.layers as tfkl

from typing import Any


class Discriminator(tfkl.Layer):
    def __init__(
        self, input_nc: int, ndf: int = 64, n_layers: int = 3, norm_layer: Any =tfkl.BatchNormalization, name="discriminator", **kwargs,
    ):
        super(Discriminator, self).__init__(name=name, **kwargs)
        """PatchGAN discriminator
        
        Inputs:
            input_nc: The number of channels in input images.
            ndf: The number of filters in the last conv layer.
            n_layers: The number of conv layers in the discriminator.
            norm_layer: Normalization layer.
        """

        kernel_size = 4
        use_bias = False  # This parameter depends on the normalisation used.

        sequence = [tfkl.Conv2D(
            filters=input_nc, kernel_size=kernel_size, strides=2, padding='same', activation=tfkl.LeakyReLU(alpha=0.2)
        )]

        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult = min(2 ** n, 8)  # TODO: Check where to start here.
            sequence += [
                tfkl.Conv2D(filters=ndf * nf_mult, kernel_size=kernel_size, strides=2, padding="same", use_bias=use_bias),
                norm_layer(ndf * nf_mult),
                tfkl.LeakyReLU(alpha=0.2),
            ]

        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            tfkl.Conv2D(filters=ndf * nf_mult, kernel_size=kernel_size, strides=1, padding="same",
                        use_bias=use_bias),
            norm_layer(ndf * nf_mult),
            tfkl.LeakyReLU(alpha=0.2),
        ]

        sequence += [tfkl.Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding="same")] # output 1 channel prediction map
        self.model = tf.keras.Sequential(layers=sequence)

    def call(self, inputs):

        # TODO: add concatenation

        return self.model(inputs)
