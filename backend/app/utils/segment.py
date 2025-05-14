import numpy as np
import cv2

def split_sentence_image_into_words(image, threshold=128, min_white_columns=20):
    """
    Splits a sentence-level image into individual word images based on vertical whitespace.

    Args:
        image (np.ndarray): Grayscale image.
        threshold (int): Binarization threshold.
        min_white_columns (int): Minimum width of whitespace to count as a word gap.

    Returns:
        List[np.ndarray]: List of word-level image segments.
    """
    if image is None:
        return []

    # Binarize
    binary_image = (image > threshold).astype(np.uint8)
    white_columns = np.all(binary_image == 1, axis=0)
    white_column_indices = np.where(white_columns)[0]

    splits = []
    current_start = None

    for i in range(len(white_column_indices)):
        if current_start is None:
            current_start = white_column_indices[i]
        if i == len(white_column_indices) - 1 or white_column_indices[i + 1] != white_column_indices[i] + 1:
            length = white_column_indices[i] - current_start + 1
            if length > min_white_columns:
                splits.append((current_start, white_column_indices[i]))
            current_start = None

    split_images = []
    last_end = 0

    for start, end in splits:
        if start > last_end:
            segment = image[:, last_end:start]
            if segment.shape[1] > 5:  # avoid narrow trash
                split_images.append(segment)
        last_end = end + 1

    if last_end < binary_image.shape[1]:
        segment = image[:, last_end:]
        if segment.shape[1] > 5:
            split_images.append(segment)

    return split_images
