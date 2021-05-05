import tensorflow as tf
from encoder import Encoder, Sampler


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        """Anna: fill in the None"""
        self.encoder = None
        self.generator = None
        self.discriminator = None
        """Alex: Sampler and putting  everything together"""
        self.sampler = Sampler()

    def generator_loss(self):
        """Seb"""
        pass

    def discriminator_loss(self, real_img):
        """Seb"""
        pass

    def kl_divergence_loss(self):
        """Anna"""
        pass

    def train_step(self):
        """Pancho"""
        pass
