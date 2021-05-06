import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_addons as tfa


class Encoder(tfkl.Layer):
    def __init__(self, name = "encoder", **kwargs):
        super(Encoder, self).__init__(name = name, **kwargs)

        # def build(self, input_shape):
        print("ssss")
        self.conv1 = tfkl.Conv2D(64, kernel_size = 3, strides = (2, 2), padding = 'same')
        self.in1 = tfa.layers.InstanceNormalization(axis = 3)
        self.conv2 = tfkl.Conv2D(128, kernel_size = 3, strides = (2, 2), padding = 'same')
        self.in2 = tfa.layers.InstanceNormalization(axis = 3)
        self.conv3 = tfkl.Conv2D(256, kernel_size = 3, strides = (2, 2), padding = 'same')
        self.in3 = tfa.layers.InstanceNormalization(axis = 3)
        self.conv4 = tfkl.Conv2D(512, kernel_size = 3, strides = (2, 2), padding = 'same')
        self.in4 = tfa.layers.InstanceNormalization(axis = 3)
        self.conv5 = tfkl.Conv2D(512, kernel_size = 3, strides = (2, 2), padding = 'same')
        self.in5 = tfa.layers.InstanceNormalization(axis = 3)
        self.conv6 = tfkl.Conv2D(512, kernel_size = 3, strides = (2, 2), padding = 'same')
        self.in6 = tfa.layers.InstanceNormalization(axis = 3)

        self.reshape = tfkl.Flatten()

        self.linear_mu = tfkl.Dense(256, activation = tfkl.LeakyReLU(alpha = 0.2))
        self.linear_var = tfkl.Dense(256, activation = tfkl.LeakyReLU(alpha = 0.2))

        self.lrelu = tfkl.LeakyReLU(alpha = 0.2)

    # def call(self, inputs, training=None):
    def call(self, input_tensor, **kwargs):
        x = tf.image.resize(input_tensor, (256, 256))
        x = self.lrelu(self.in1(self.conv1(x)))
        x = self.lrelu(self.in2(self.conv2(x)))
        x = self.lrelu(self.in3(self.conv3(x)))
        x = self.lrelu(self.in4(self.conv4(x)))
        x = self.lrelu(self.in5(self.conv5(x)))
        x = self.lrelu(self.in6(self.conv6(x)))

        x = self.reshape(x)

        out_mu = self.linear_mu(x)
        out_var = self.linear_var(x)

        return out_mu, out_var


class Sampler(tfkl.Layer):
    def call(self, inputs, train = True):
        # print("Inputs ", inputs.shape)
        z_mean, z_log_var = inputs

        if train:
            epsilon = tf.keras.backend.random_normal(shape = tf.shape(z_log_var))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        else:
            return z_mean
