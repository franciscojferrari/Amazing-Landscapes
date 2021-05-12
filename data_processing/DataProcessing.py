import time

from data_processing.utils import *


class DataWriter:
    def __init__(self, bucket_name, config, set_type):
        self.bucket_path = bucket_name
        self.config = config

        self.set_type = set_type
        self.img_height = self.config["img_height"]
        self.img_width = self.config["img_width"]

        self.processed_files = None

    def process_files_cityscape(self) -> None:
        """Process files that we wish to write to a dataset."""
        # get all files
        file_names = find_files_cityscape(set_type = self.set_type)

        # Create a dataset containing the filenames
        file_dataset = build_file_dataset(file_names)

        # Apply the processing function to the files
        self.processed_files = transform_files(file_dataset, self.load_and_downsample_cityscape)

    def process_files_celebmask(self) -> None:
        file_names = find_files_celebmask()
        file_dataset = build_file_dataset(file_names)
        self.processed_files = transform_files(file_dataset, self.load_and_downsample_celebmask)

    def write_files_cityscape(self) -> None:
        start_time = time.time()
        print("Start writing files")

        record_file_name = f"{self.set_type}.tfrecords"
        write_path = f"{self.bucket_path}/cityscape/processed_data"
        record_file = os.path.join(write_path, record_file_name)
        n_samples = tf.data.experimental.cardinality(self.processed_files).numpy()
        print(f"Number of samples in dataset: {n_samples}")

        with tf.io.TFRecordWriter(record_file) as writer:
            for img_original, img_masked, label in self.processed_files:
                tf_example = datapoint_example(
                    img_original, img_masked, label, self.set_type
                )
                writer.write(tf_example.SerializeToString())

        print(f"Finished writing files in:  {time.time() - start_time}s")

    def write_files_celebmask(self):
        start_time = time.time()
        print("Start writing files")

        record_file_name = "dataset.tfrecords"
        write_path = f"{self.bucket_path}/CelebAMask/processed_data"
        record_file = os.path.join(write_path, record_file_name)
        n_samples = tf.data.experimental.cardinality(self.processed_files).numpy()
        print(f"Number of samples in dataset: {n_samples}")

        with tf.io.TFRecordWriter(record_file) as writer:
            for img_original, img_masked, label in self.processed_files:
                tf_example = datapoint_example(
                    img_original, img_masked, label, self.set_type
                )
                writer.write(tf_example.SerializeToString())

        print(f"Finished writing files in:  {time.time() - start_time}s")

    def load_and_downsample_cityscape(self, file_path: tf.Tensor) -> [tf.Tensor, tf.Tensor, str]:
        """Load image file from disk and perform downsampling"""

        file_path = file_path.numpy().decode("utf-8")
        label = get_label(file_path)

        masked_img_path = (
            f"{self.bucket_path}/cityscape/gtfine/gtFine/{file_path}_gtFine_color.png"
        )
        original_img_path = f"{self.bucket_path}/cityscape/trainvaltest/leftImg8bit/{file_path}_leftImg8bit.png"

        # Process original img
        img_original = load_img(original_img_path)
        img_original = tf.image.resize(
            img_original,
            size = [self.img_height, self.img_width],
            method = "nearest",
            preserve_aspect_ratio = self.config["preserve_aspect_ratio"],
        )

        # Process masked img
        img_masked = load_img(masked_img_path)
        img_masked = tf.image.resize(
            img_masked,
            size = [self.img_height, self.img_width],
            method = "nearest",
            preserve_aspect_ratio = self.config["preserve_aspect_ratio"],
        )

        return img_original, img_masked, label

    def load_and_downsample_celebmask(self, file_name: tf.Tensor) -> [tf.Tensor, tf.Tensor, str]:

        label = file_name.split(".")[0]
        original_img_path = f"{self.bucket_path}/CelebAMask/CelebA-HQ-img/{file_name}.jpg"
        masked_img_path = f"{self.bucket_path}/CelebAMask/CelebAMask-HQ-mask-color/{file_name}.png"

        # Process original img
        img_original = load_img(original_img_path)
        img_original = tf.image.resize(
            img_original,
            size=[self.img_height, self.img_width],
            method="nearest",
            preserve_aspect_ratio=self.config["preserve_aspect_ratio"],
        )

        # Process masked img
        img_masked = load_img(masked_img_path)
        img_masked = tf.image.resize(
            img_masked,
            size=[self.img_height, self.img_width],
            method="nearest",
            preserve_aspect_ratio=self.config["preserve_aspect_ratio"],
        )

        return img_original, img_masked, label


class DataReader:
    def __init__(self, base_path, set_type, config):
        self.base_path = base_path
        self.set_type = set_type
        self.config = config

        self.feature_description = {
            "label": tf.io.FixedLenFeature([], tf.string),
            "img_original": tf.io.RaggedFeature(tf.string),
            "img_masked": tf.io.RaggedFeature(tf.string),
            "subset": tf.io.FixedLenFeature([], tf.string),
        }

        self.data_set = None

    def _parse_function(self, example_proto):
        # Parse the input tf.train.Example proto using the provided dictionary
        return tf.io.parse_single_example(example_proto, self.feature_description)

    def read_data_set(self) -> None:
        """Read image dataset and parse it."""
        tfrecord_path = os.path.join(self.base_path, f"{self.set_type}.tfrecords")
        processed_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_dataset = processed_dataset.map(self._parse_function)
        self.data_set = parsed_dataset.map(self.parse_serialized_tensors)

    def parse_serialized_tensors(self, example):
        """Convert the serialized tensor back to a tensor."""
        example["img_original"] = tf.reshape(example["img_original"], [])
        example["img_masked"] = tf.reshape(example["img_masked"], [])

        img_height = self.config['img_height']
        img_width = self.config['img_width']
        example["img_original"] = tf.ensure_shape(tf.io.parse_tensor(example["img_original"], out_type = tf.float32),
                                                  (img_height, img_width, 3))
        example["img_masked"] = tf.ensure_shape(tf.io.parse_tensor(example["img_masked"], out_type = tf.float32),
                                                (img_height, img_width, 3))
        return example

    def get_dataset(self) -> tf.data.TFRecordDataset:
        """Getter for the dataset."""
        return self.data_set
