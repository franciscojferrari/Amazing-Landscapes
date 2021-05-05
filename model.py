import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        """Anna: fill in the None"""
        self.encoder = None
        self.generator = None
        self.discriminator = None
        """Alex: Sampler and putting  everything together"""
        self.sampler = None

    def generator_loss(self):
        """Seb"""
        pass

    def discriminator_loss(self, real_img):
        """Seb"""
        pass

    def kl_divergence_loss(self):
        """Anna"""
        pass

    def train_step(self, data):
        images, masks = data
        with tf.GradientTape() as tape:
            # Forward pass
            image_style_mu_var = self.encoder(images)
            z_noise_style = self.sampler(image_style_mu_var)
            generated_image = self.generator(masks, z_noise = z_noise_style)

            real_output = self.discriminator(images, masks)
            fake_output = self.discriminator(generated_image, masks)
            