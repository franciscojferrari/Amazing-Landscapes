import tensorflow as tf
import tensorflow_addons as tfa

tfk = tf.keras
tfkl = tfk.layers


class Spade(tfkl.Layer):
    def __init__(self, name, n_out_filter, **kwargs):
        super(Spade, self).__init__(name = name, **kwargs)
        # TODO: Check if other layers should have relu activation
        # TODO: Maybe add parameterized kernel size and padding. SEE: Paper's github
        self.bn = tfkl.BatchNormalization(momentum = 0.9, epsilon = 1e-5, center = False, scale = False)
        self.conv0 = tfa.layers.SpectralNormalization(tfkl.Conv2D(128, 3, activation = "relu", padding = "same"))
        self.conv1 = tfa.layers.SpectralNormalization(tfkl.Conv2D(n_out_filter, 3, padding = "same"))
        self.conv2 = tfa.layers.SpectralNormalization(tfkl.Conv2D(n_out_filter, 3, padding = "same"))

    def call(self, features, *args, **kwargs):
        mask = kwargs.get("mask", args[0] if args else None)

        size = features.shape[1:3]
        mask = tf.image.resize(mask, size, method = "nearest")
        interim_conv = self.conv0(mask)
        gamma = self.conv1(interim_conv)
        beta = self.conv2(interim_conv)

        return ((self.bn(features) * gamma) + beta)


class SpadeResidualBlock(tfkl.Layer):
    def __init__(self, name, n_input_filter, n_out_filter, **kwargs):
        super(SpadeResidualBlock, self).__init__(name = name, **kwargs)

        self.learned_shortcut = (n_input_filter != n_out_filter)

        # TODO: Possible to add input dimensions for debugging and testing

        self.LeakyReLU = tfkl.LeakyReLU(alpha = 2e-1)

        self.spade0 = Spade(name = f"{name}_spade0", n_out_filter = n_input_filter)

        self.conv0 = tfa.layers.SpectralNormalization(tfkl.Conv2D(n_out_filter, 3))

        self.spade1 = Spade(name = f"{name}_spade1", n_out_filter = n_out_filter)
        self.conv1 = tfa.layers.SpectralNormalization(tfkl.Conv2D(n_out_filter, 3))

        if self.learned_shortcut:
            self.spade_skip = Spade(name = f"{name}_spade1", n_out_filter = n_input_filter)
            self.conv_skip = tfa.layers.SpectralNormalization(tfkl.Conv2D(n_out_filter, 3))

    def call(self, features, *args, **kwargs):
        mask = kwargs.get("mask", args[0] if args else None)

        skip_features = features
        if self.learned_shortcut:
            skip_features = self.conv_skip(self.LeakyReLU(self.spade_skip(features, mask = mask)))

        features = self.conv0(self.LeakyReLU(self.spade0(features, mask = mask)))
        features = self.conv1(self.LeakyReLU(self.spade1(features, mask = mask)))

        return features + skip_features
