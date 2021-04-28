import tensorflow as tf

from layers import SpadeResidualBlock

# import tensorflow_addons as tfa

tfk = tf.keras
tfkl = tfk.layers


class SpadeGenerator(tfkl.Layer):
    def __init__(self, name, **kwargs):
        super(SpadeGenerator).__init__(name = name, **kwargs)
        self.input_noise_dim = kwargs.get("input_noise_dim", None)
        self.output_dim = kwargs.get("output_dim", None)
        self.batch_size = kwargs.get("batch_size", None)
        self.image_size = kwargs.get("image_size", (None, None))  # This should be images size

        self.sw, self.sh = self.compute_latent_vector_size()
        self.z_dim = 256
        self.nf = 64

        self.linear_layer_0 = tfkl.Dense(16 * self.nf * self.sw * self.sh)

        self.spade_0 = SpadeResidualBlock(16 * self.nf, 16 * self.nf)
        self.spade_1 = SpadeResidualBlock(16 * self.nf, 16 * self.nf)
        self.spade_2 = SpadeResidualBlock(16 * self.nf, 16 * self.nf)
        self.spade_3 = SpadeResidualBlock(16 * self.nf, 8 * self.nf)
        self.spade_4 = SpadeResidualBlock(8 * self.nf, 4 * self.nf)
        self.spade_5 = SpadeResidualBlock(4 * self.nf, 2 * self.nf)
        self.spade_6 = SpadeResidualBlock(2 * self.nf, 1 * self.nf)

        self.conv_7 = tfkl.Conv2D(3, 3, padding = "same")
        self.LeakyReLU = tfkl.LeakyReLU(alpha = 2e-1)

        self.up_sample = tfkl.UpSampling2D(2)

    def compute_latent_vector_size(self):
        num_up_layers = 5
        aspect_ratio = 1

        sw = self.image_size[0] // (2 ** num_up_layers)
        sh = round(sw / aspect_ratio)

        return sw, sh

    def __call__(self, *args, **kwargs):
        mask = kwargs.get("mask", args[0])
        z_noise = kwargs.get("z_noise", None)

        if z_noise is None:
            z_noise = tf.random.normal([mask.shape[0], self.z_dim], 0, 1, dtype = tf.float32)

        x = self.linear_layer_0(z_noise)
        x = tfkl.Reshape(-1, 16 * self.nf, self.sh, self.sw)

        x = self.up_sample(self.spade_0(features = x, mask = mask))
        x = self.up_sample(self.spade_1(features = x, mask = mask))
        x = self.up_sample(self.spade_2(features = x, mask = mask))
        x = self.up_sample(self.spade_3(features = x, mask = mask))
        x = self.up_sample(self.spade_4(features = x, mask = mask))
        x = self.up_sample(self.spade_5(features = x, mask = mask))
        x = self.up_sample(self.spade_6(features = x, mask = mask))

        x = self.conv_7(self.LeakyReLU(x))
        x = tf.keras.activations.tanh(x)

        return x
