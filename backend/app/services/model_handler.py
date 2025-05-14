import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from app.utils.ctc_layer import CTCLayer
from app.utils.segment import split_sentence_image_into_words
from app.utils.preprocess import preprocess_image
from app.utils.decode import decode_predictions
from app.utils.config import max_len  # Ensure this is defined correctly

def load_handwriting_model():
    # Load the model with the custom CTC layer registered
    return keras_load_model(
        "app/models/handwriting_model.h5",
        compile=False,
        custom_objects={"CTCLayer": CTCLayer}
    )

def predict_sentence(image_file, model):
    # Decode uploaded image file
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Segment into individual word images
    word_imgs = split_sentence_image_into_words(img)
    words = []

    for word_img in word_imgs:
        # Preprocess each word image to shape (128, 32, 1)
        processed = preprocess_image(word_img)
        processed = tf.expand_dims(processed, 0)  # Shape: (1, 128, 32, 1)

        # Dummy label to satisfy model input
        dummy_label = tf.zeros((1, max_len), dtype=tf.int64)

        # Predict and decode
        pred = model.predict({"image": processed, "label": dummy_label})
        word = decode_predictions(pred)
        words.append(word[0])

    return " ".join(words)
