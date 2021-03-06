import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


class VGGLoss(tf.keras.Model):
    def __init__(self, name = "vggloss", **kwargs):
        super(VGGLoss, self).__init__(name = name, **kwargs)
        self.vgg = Vgg19()
        self.layer_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    # def call(self, inputs, training=None):
    def call(self, inputs, **kwargs):
        x, y = inputs

        x = (x + 1) * 127.5
        y = (y + 1) * 127.5
        x_vgg, y_vgg = self.vgg(preprocess_input(x)), self.vgg(preprocess_input(y))

        loss = 0

        for i in range(len(x_vgg)):
            y_vgg_detach = tf.stop_gradient(y_vgg[i])
            loss += self.layer_weights[i] * L1_loss(x_vgg[i], y_vgg_detach)

        return loss


class Vgg19(tf.keras.Model):
    def __init__(self, trainable = False):
        super(Vgg19, self).__init__(name = 'Vgg19')
        vgg_pretrained_features = tf.keras.applications.vgg19.VGG19(weights = 'imagenet', include_top = False)

        if trainable is False:
            vgg_pretrained_features.trainable = False

        vgg_pretrained_features = vgg_pretrained_features.layers

        self.slice1 = tf.keras.Sequential()
        self.slice2 = tf.keras.Sequential()
        self.slice3 = tf.keras.Sequential()
        self.slice4 = tf.keras.Sequential()
        self.slice5 = tf.keras.Sequential()

        for x in range(1, 2):
            self.slice1.add(vgg_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add(vgg_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add(vgg_pretrained_features[x])
        for x in range(8, 13):
            self.slice4.add(vgg_pretrained_features[x])
        for x in range(13, 18):
            self.slice5.add(vgg_pretrained_features[x])

    def call(self, inputs, **kwargs):
        h_relu1 = self.slice1(inputs)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
