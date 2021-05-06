import tensorflow as tf

from .Discriminator import Discriminator
from .SpadeGenerator import SpadeGenerator
from .encoder import Encoder, Sampler


class Model(tf.keras.Model):
    def __init__(self, name = "model", **kwargs):
        super(Model, self).__init__(name = name, **kwargs)

        """Anna: fill in the None"""
        self.encoder = Encoder()
        self.generator = SpadeGenerator(image_size = (256, 256))
        # print("Generator")
        self.discriminator = Discriminator()
        # """Alex: Sampler and putting  everything together"""
        self.sampler = Sampler()

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

    def kl_divergence_loss(self, mu, logvar):
        return -0.5 * tf.math.reduce_sum(1 + logvar - tf.math.pow(mu, 2) - tf.math.exp(logvar))

    def train_step(self, data):
        images, masks = data[0]
        with tf.GradientTape() as generator_tape:
            # Forward pass
            out_mu, out_var = self.encoder(images)
            z_noise_style = self.sampler((out_mu, out_var))
            generated_image = self.generator(masks, z_noise = z_noise_style)

            fake_output = self.discriminator(generated_image, masks)

            generator_loss = self.loss_generator(fake_output)

            kl_loss = self.kl_divergence_loss(out_mu, out_var)
            total_generator_loss = kl_loss + generator_loss

        generator_gradients = generator_tape.gradient(
            total_generator_loss, (self.generator.trainable_variables + self.encoder.trainable_variables)
        )
        self.optimizer.apply_gradients(
            zip(generator_gradients, (self.generator.trainable_variables + self.encoder.trainable_variables)))

        with tf.GradientTape() as discriminator_tape:
            # Forward pass
            out_mu, out_var = self.encoder(images)
            z_noise_style = self.sampler((out_mu, out_var))
            generated_image = self.generator(masks, z_noise = z_noise_style)

            real_output = self.discriminator(images, masks)
            fake_output = self.discriminator(generated_image, masks)

            discriminator_loss = self.loss_discriminator(real_output, fake_output)

        discriminator_gradients = discriminator_tape.gradient(
            discriminator_loss, self.discriminator.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        return {"total_generator_loss": total_generator_loss, "generator_loss": generator_loss, "kl_loss": kl_loss,
                "discriminator_loss": discriminator_loss}
