import tensorflow as tf

from .layers import SpadeResidualBlock

# import tensorflow_addons as tfa

tfk = tf.keras
tfkl = tfk.layers


class SpadeGenerator(tfkl.Layer):
    def __init__(self, name, image_size, **kwargs):
        super(SpadeGenerator, self).__init__(name = name, **kwargs)

        self.image_size = image_size
        self.sw, self.sh = self.compute_latent_vector_size()
        self.z_dim = 256
        self.nf = 64

        self.linear_layer_0 = tfkl.Dense(16 * self.nf * self.sw * self.sh)

        self.spade_0 = SpadeResidualBlock("spade_0", 16 * self.nf, 16 * self.nf)
        self.spade_1 = SpadeResidualBlock("spade_1", 16 * self.nf, 16 * self.nf)
        self.spade_2 = SpadeResidualBlock("spade_2", 16 * self.nf, 16 * self.nf)
        self.spade_3 = SpadeResidualBlock("spade_3", 16 * self.nf, 8 * self.nf)
        self.spade_4 = SpadeResidualBlock("spade_4", 8 * self.nf, 4 * self.nf)
        self.spade_5 = SpadeResidualBlock("spade_5", 4 * self.nf, 2 * self.nf)
        self.spade_6 = SpadeResidualBlock("spade_6", 2 * self.nf, 1 * self.nf)

        self.conv_7 = tfkl.Conv2D(3, 3, padding = "same")
        self.LeakyReLU = tfkl.LeakyReLU(alpha = 2e-1)

        self.up_sample = tfkl.UpSampling2D(2)

    def compute_latent_vector_size(self):
        num_up_layers = 6
        aspect_ratio = 1

        sw = self.image_size[0] // (2 ** num_up_layers)
        sh = round(sw / aspect_ratio)

        return sw, sh

    def __call__(self, mask, *args, **kwargs):
        z_noise = kwargs.get("z_noise", args[0] if args else None)

        if z_noise is None:
            z_noise = tf.random.normal([mask.shape[0], self.z_dim], 0, 1, dtype = tf.float32)

        x = self.linear_layer_0(z_noise)
        print("linear ", x.shape)
        # x = tfkl.Reshape((-1, 16 * self.nf, self.sh, self.sw))(x)
        x = tf.reshape(x, [-1, self.sh, self.sw, 16 * self.nf])
        print("reshape ", x.shape)
        x = self.up_sample(self.spade_0(features = x, mask = mask))
        print("spade 1 - upsample ", x.shape)
        x = self.up_sample(self.spade_1(features = x, mask = mask))
        print("spade 2 - upsample ", x.shape)
        x = self.up_sample(self.spade_2(features = x, mask = mask))
        print("spade 3 - upsample ", x.shape)
        x = self.up_sample(self.spade_3(features = x, mask = mask))
        print("spade 4 - upsample ", x.shape)
        x = self.up_sample(self.spade_4(features = x, mask = mask))
        print("spade 5 - upsample ", x.shape)
        x = self.up_sample(self.spade_5(features = x, mask = mask))
        print("spade 6 - upsample ", x.shape)
        x = self.up_sample(self.spade_6(features = x, mask = mask))
        print("spade 7 - upsample ", x.shape)

        x = self.conv_7(self.LeakyReLU(x))
        print("conv ", x.shape)
        x = tf.keras.activations.tanh(x)

        return x
