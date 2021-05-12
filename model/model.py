import tensorflow as tf

from .Discriminator import Discriminator
from .SpadeGenerator import SpadeGenerator
from .encoder import Encoder, Sampler
from .vgg19 import VGGLoss


class LearningRateReducer(tf.keras.callbacks.Callback):
    def __init__(self, decay_epoch, total_epochs):
        super(LearningRateReducer, self).__init__()
        self.decay_epoch = decay_epoch
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs = {}):

        if epoch == 1:
            self.init_lr_g = self.model.g_optimizer.lr.read_value()
            self.init_lr_d = self.model.d_optimizer.lr.read_value()
            print("Epoch: ", epoch)

        if epoch > self.decay_epoch:
            new_lr_g = self.init_lr_g * (self.total_epochs - epoch) / (self.total_epochs - self.decay_epoch)
            self.model.g_optimizer.lr.assign(new_lr_g)

            new_lr_d = self.init_lr_d * (self.total_epochs - epoch) / (self.total_epochs - self.decay_epoch)
            self.model.d_optimizer.lr.assign(new_lr_d)


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
        self.vgg_loss = VGGLoss()

    def loss_discriminator(self, real, fake, loss_type = None):
        if loss_type == "hinge":
            # Real loss
            min_val = tf.math.minimum(real - 1, tf.zeros_like(real))
            real_loss = - tf.reduce_mean(min_val)
            # Fake loss.
            min_val = tf.math.minimum(-fake - 1, tf.zeros_like(fake))
            fake_loss = - tf.reduce_mean(min_val)

        elif loss_type == "gan":
            cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
            real_loss = cross_entropy(tf.ones_like(real), real)
            fake_loss = cross_entropy(tf.zeros_like(fake), fake)
        return real_loss + fake_loss

    def loss_generator(self, fake, loss_type = None):
        if loss_type == "hinge":
            return - tf.reduce_mean(fake)
        elif loss_type == "gan":
            cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
            return cross_entropy(tf.ones_like(fake), fake)

    def g_feature_loss(self, fake, target):
        l1_loss = tf.reduce_mean(tf.abs(fake - target))
        return l1_loss

    def kl_divergence_loss(self, mu, logvar):
        return -0.5 * tf.math.reduce_sum(1 + logvar - tf.math.pow(mu, 2) - tf.math.exp(logvar))

    def compile(self, d_optimizer, g_optimizer, metrics = None):
        super(Model, self).compile(metrics = metrics)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, data):
        images = data['img_original']
        masks = data['img_masked']

        with tf.GradientTape() as generator_tape:
            # Forward pass
            out_mu, out_var = self.encoder(images)
            z_noise_style = self.sampler((out_mu, out_var))
            generated_image = self.generator(masks, z_noise = z_noise_style)

            fake_output = self.discriminator(generated_image, masks)
            real_output = self.discriminator(images, masks)

            generator_loss = self.loss_generator(fake_output, loss_type = "hinge")
            kl_loss = self.kl_divergence_loss(out_mu, out_var)
            vgg_loss = self.vgg_loss((images, generated_image))
            g_feature_loss = self.g_feature_loss(fake_output, real_output)

            vgg_weight = 2
            lambda_ = 0.05
            g_feature_loss_lambda = 5

            total_generator_loss = tf.math.scalar_mul(lambda_, kl_loss) \
                                   + generator_loss \
                                   + tf.math.scalar_mul(vgg_weight, vgg_loss) \
                                   + tf.math.scalar_mul(g_feature_loss_lambda, g_feature_loss)

        generator_gradients = generator_tape.gradient(
            total_generator_loss, (self.generator.trainable_variables + self.encoder.trainable_variables)
        )
        self.g_optimizer.apply_gradients(
            zip(generator_gradients, (self.generator.trainable_variables + self.encoder.trainable_variables)))

        with tf.GradientTape() as discriminator_tape:
            # Forward pass
            out_mu, out_var = self.encoder(images)
            z_noise_style = self.sampler((out_mu, out_var))
            generated_image = self.generator(masks, z_noise = z_noise_style)

            real_output = self.discriminator(images, masks)
            fake_output = self.discriminator(generated_image, masks)

            discriminator_loss = self.loss_discriminator(real_output, fake_output, loss_type = "hinge")

        discriminator_gradients = discriminator_tape.gradient(
            discriminator_loss, self.discriminator.trainable_variables
        )
        self.d_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        return {"total_generator_loss": total_generator_loss, "generator_loss": generator_loss,
                "g_feature_loss": g_feature_loss, "kl_loss": kl_loss,
                "discriminator_loss": discriminator_loss, "vgg_loss": vgg_loss,
                "lr_d": self.d_optimizer.lr, "lr_g": self.g_optimizer.lr}
