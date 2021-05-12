import os
from typing import Any, List

import tensorflow as tf


def find_files_cityscape(set_type = "train"):
    # set_type can be "train", "test" or "val"
    file_names = []

    path = f"DataSet/cityscape/gtfine/gtFine/{set_type}"

    cities = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    for city in cities:
        city_path = os.path.join(path, city)

        # get all files per city
        for file in os.listdir(city_path):
            if file.endswith("color.png"):
                file_path = os.path.join(city_path, file)
                file_path = file_path.split("_gtFine_")[0]
                file_path = file_path.split(f"/gtFine/")[-1]
                file_names.append(file_path)

    return file_names


def find_files_celebmask(nr_files):
    return [f"{i}" for i in range(nr_files)]


def build_file_dataset(file_paths: List[str]) -> tf.data.Dataset:
    """Create a DataSet object containing all file paths"""
    file_paths = tf.convert_to_tensor(file_paths, dtype = tf.string)
    dataset = tf.data.Dataset.from_tensor_slices(file_paths).prefetch(1)
    return dataset


def transform_files(files: tf.data.Dataset, process_fn: Any) -> tf.data.Dataset:
    """Apply processing step to dataset"""
    files = files.map(
        lambda x: tf.py_function(process_fn, [x], [tf.uint8, tf.uint8, tf.string]),
        num_parallel_calls = tf.data.experimental.AUTOTUNE,
    )

    return files


def get_label(file_path: str) -> str:
    """Extract label from file path."""
    return file_path.split("/")[-1]


def load_img(file_path: str) -> tf.Tensor:
    """Load and decode image."""
    img = tf.io.read_file(file_path)

    extension = os.path.splitext(file_path)[1]

    if extension == ".png":
        return tf.image.decode_png(img, channels = 3)

    elif extension == ".jpeg" or extension == ".jpg":
        return tf.image.decode_jpeg(img, channels = 3)


def _bytes_feature(value: Any) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))


def normalize_images(images, masks):
    images = (tf.cast(images, dtype = tf.float32) / 127.5) - 1.0
    masks = (tf.cast(masks, dtype = tf.float32) / 127.5) - 1.0
    return images, masks


# Create a dictionary with features that may be relevant.
def datapoint_example(
        img_orignal: tf.Tensor, img_masked: tf.Tensor, label: int, subset: str
) -> tf.train.Example:
    img_orignal, img_masked = normalize_images(img_orignal, img_masked)
    img_original_serialized = tf.io.serialize_tensor(img_orignal)
    img_masked_serialized = tf.io.serialize_tensor(img_masked)
    feature = {
        "label": _bytes_feature(label),
        "img_original": _bytes_feature(img_original_serialized),
        "img_masked": _bytes_feature(img_masked_serialized),
        "subset": _bytes_feature(str.encode(subset)),
    }
    return tf.train.Example(features = tf.train.Features(feature = feature))

