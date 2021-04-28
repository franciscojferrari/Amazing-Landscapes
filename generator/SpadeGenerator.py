import tensorflow as tf

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

        tfkl.Dense()

    def compute_latent_vector_size(self):
        num_up_layers = 5
        aspect_ratio = 1

        sw = self.image_size[0] // (2 ** num_up_layers)
        sh = round(sw / aspect_ratio)

        return sw, sh
