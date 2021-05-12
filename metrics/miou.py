import tensorflow as tf


def denormalize_images(images):
    images = tf.convert_to_tensor(images)
    images = tf.cast((images + 1.0) * 127.5, dtype=tf.int32)
    return images


def calculate_miou(prediction, ground_truth):
    """Calculate Mean IoU based on prediction and ground truth masks.

    Input:
      prediction: Predicted masks (can be a list, np.array or Tensor)
      ground_truth: Ground truth masks (can be a list, np.array or Tensor)

    Returns:
      Mean IoU score
    """

    # Denormalize images
    prediction = denormalize_images(prediction)
    ground_truth = denormalize_images(ground_truth)

    # Convert to grayscale images
    prediction = tf.image.rgb_to_grayscale(prediction)
    ground_truth = tf.image.rgb_to_grayscale(ground_truth)

    # Calculate MeanIoU score.
    m = tf.keras.metrics.MeanIoU(num_classes=255)  # TODO check if this matters
    m.update_state(prediction, ground_truth)
    score = m.result().numpy()

    return score
