import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_addons as tfa


class Discriminator(tfkl.Layer):
    def __init__(
            self,
            name = "discriminator",
            **kwargs,
    ):
        super(Discriminator, self).__init__(name = name, **kwargs)

    def build(self, input_shape):
        """PatchGAN discriminator"""
        self.leakyrelu = tfkl.LeakyReLU(alpha = 0.2)
        self.layer1 = tfkl.Conv2D(
            filters = 64,
            kernel_size = 4,
            strides = 2,
            padding = "same",
            activation = tfkl.LeakyReLU(alpha = 0.2),
        )

        self.layer2 = tfkl.Conv2D(
            filters = 128,
            kernel_size = 4,
            strides = 2,
            padding = "same",
        )
        self.norm2 = tfa.layers.InstanceNormalization(groups = 128)

        self.layer3 = tfkl.Conv2D(
            filters = 256,
            kernel_size = 4,
            strides = 2,
            padding = "same",
        )
        self.norm3 = tfa.layers.InstanceNormalization(groups = 256)

        self.layer4 = tfkl.Conv2D(
            filters = 512,
            kernel_size = 4,
            strides = 2,
            padding = "same",
        )
        self.norm4 = tfa.layers.InstanceNormalization(groups = 512)

        self.output_layer = tfkl.Conv2D(
            filters = 1,
            kernel_size = 4,
            strides = 1,
            padding = "same")

        self.down_sampler = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)

    def loss_discriminator(self, real, fake):

        # Real loss
        min_val = tf.math.minimum(real - 1, tf.zeros_like(real))
        real_loss = - tf.reduce_mean(min_val)

        # Fake loss.
        min_val = tf.math.minimum(-fake - 1, tf.zeros_like(fake))
        fake_loss = - tf.reduce_mean(min_val)

        return real_loss + fake_loss

    def loss_generator(self, fake):
        return - tf.reduce_mean(fake)

    def total_loss(self, real_images, real_masks, fake_images, fake_masks):
        """Implements the hinge loss function for both generator and discriminator."""

        real = self.call(real_images, real_masks)
        fake = self.call(fake_images, fake_masks)

        generator_loss = self.loss_generator(fake)
        discriminator_loss = self.loss_discriminator(real, fake)

        return generator_loss + discriminator_loss

    def call(self, image, mask, down=2):

        final = tf.zeros(image.shape[0])
        for i in range(down + 1):
            if i != 0:
                # Down sample img and mask.
                image = self.down_sampler(image)
                mask = self.down_sampler(mask)

            inputs = tf.concat([image, mask], axis=-1)

            x = self.layer1(inputs)
            x = self.leakyrelu(self.norm2(self.layer2(x)))
            x = self.leakyrelu(self.norm3(self.layer3(x)))
            x = self.leakyrelu(self.norm4(self.layer4(x)))

            out = self.output_layer(x)

            final = tf.math.add(final, tf.math.reduce_mean(out, axis=(1, 2, 3)))

        return final
