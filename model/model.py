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
    def __init__(self, name = "model", *args, **kwargs):
        self.config = kwargs.pop("config", args[0] if args else None)

        super(Model, self).__init__(name = name, **kwargs)

        """Anna: fill in the None"""
        image_size = (self.config['img_width'], self.config['img_height'])

        self.encoder = Encoder()
        self.generator = SpadeGenerator(image_size = image_size)
        # print("Generator")
        self.discriminator = Discriminator()
        # """Alex: Sampler and putting  everything together"""
        self.sampler = Sampler()
        self.vgg_loss = VGGLoss()

    def loss_discriminator(self, real, fake, loss_type = None):
        loss = []
        for i in range(len(fake)):
            if loss_type == "hinge":
                # Real loss
                min_val = tf.math.minimum(real[i][-1] - 1, tf.zeros_like(real[i][-1]))
                real_loss = - tf.reduce_mean(min_val)
                # Fake loss.
                min_val = tf.math.minimum(-fake[i][-1] - 1, tf.zeros_like(fake[i][-1]))
                fake_loss = - tf.reduce_mean(min_val)

            elif loss_type == "gan":
                cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
                real_loss = cross_entropy(tf.ones_like(real[i][-1]), real[i][-1])
                fake_loss = cross_entropy(tf.zeros_like(fake[i][-1]), fake[i][-1])
            loss.append(real_loss + fake_loss)

        return tf.reduce_mean(loss)

    def loss_generator(self, fake, loss_type = None):
        loss = []
        fake_loss = 0
        for i in range(len(fake)):
            if loss_type == "hinge":
                fake_loss = - tf.reduce_mean(fake[i][-1])
            elif loss_type == "gan":
                cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
                fake_loss = cross_entropy(tf.ones_like(fake[i][-1]), fake[i][-1])
            loss.append(fake_loss)
        return tf.reduce_mean(loss)

    def l1_loss(self, real, fake):
        return tf.reduce_mean(tf.abs(real - fake))

    def feature_loss(self, real, fake):
        loss = []
        for i in range(len(fake)):
            intermediate_loss = 0
            for j in range(len(fake[i]) - 1):
                intermediate_loss += self.l1_loss(real[i][j], fake[i][j])
            loss.append(intermediate_loss)

        return tf.reduce_mean(loss)

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

            _, fake_logit = self.discriminator(generated_image, masks)
            _, real_logit = self.discriminator(images, masks)

            generator_loss = self.loss_generator(fake_logit, loss_type = "hinge")
            kl_loss = self.kl_divergence_loss(out_mu, out_var)
            vgg_loss = self.vgg_loss((images, generated_image))
            feature_loss = self.feature_loss(real_logit, fake_logit)

            vgg_weight = self.config['vgg_weight']  # discriminator_loss5
            lambda_ = self.config['lambda_']  # 0.05
            feature_loss_lambda = self.config['feature_loss_lambda']  # 10

            total_generator_loss = tf.math.scalar_mul(lambda_, kl_loss) \
                                   + generator_loss \
                                   + tf.math.scalar_mul(vgg_weight, vgg_loss) \
                                   + tf.math.scalar_mul(feature_loss_lambda, feature_loss)
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

            _, real_logit = self.discriminator(images, masks)
            _, fake_logit = self.discriminator(generated_image, masks)

            discriminator_loss = self.loss_discriminator(real_logit, fake_logit, loss_type = "hinge")

        discriminator_gradients = discriminator_tape.gradient(
            discriminator_loss, self.discriminator.trainable_variables
        )
        self.d_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        return {"total_generator_loss": total_generator_loss, "generator_loss": generator_loss,
                "feature_loss": feature_loss, "kl_loss": kl_loss,
                "discriminator_loss": discriminator_loss, "vgg_loss": vgg_loss,
                "lr_d": self.d_optimizer.lr, "lr_g": self.g_optimizer.lr}
