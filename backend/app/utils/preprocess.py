import numpy as np
import tensorflow as tf

image_width = 128
image_height = 32

def distortion_free_resize(image, target_size):
    """
    Resize image with preserved aspect ratio and padding.
    Expects image as a TensorFlow tensor (HWC).
    """
    target_w, target_h = target_size

    # Resize while preserving aspect ratio
    image = tf.image.resize(image, size=(target_h, target_w), preserve_aspect_ratio=True)

    # Compute padding amounts
    pad_height = target_h - tf.shape(image)[0]
    pad_width = target_w - tf.shape(image)[1]

    # Symmetric padding: handle odd values
    pad_top = pad_height // 2 + (pad_height % 2)
    pad_bottom = pad_height // 2
    pad_left = pad_width // 2 + (pad_width % 2)
    pad_right = pad_width // 2

    # Pad image
    image = tf.pad(
        image,
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        constant_values=0.0  # Background is black before normalization
    )

    # Transpose to (width, height, channels)
    image = tf.transpose(image, [1, 0, 2])

    # Flip (as in notebook)
    image = tf.image.flip_left_right(image)

    return image

def preprocess_image(np_img, img_size=(image_width, image_height)):
    """
    Preprocesses a grayscale NumPy image: resizing, padding, normalization.

    Args:
        np_img (np.ndarray): Input grayscale image as NumPy array (2D).
        img_size (tuple): Target size (width, height).

    Returns:
        np.ndarray: Preprocessed image tensor (128, 32, 1)
    """
    # Normalize grayscale image and add channel dimension
    np_img = np.expand_dims(np_img, axis=-1)  # (H, W, 1)
    img_tensor = tf.convert_to_tensor(np_img, dtype=tf.float32)

    # Resize with padding
    processed = distortion_free_resize(img_tensor, img_size)

    # Normalize after resize + transpose
    processed = processed / 255.0

    return processed
