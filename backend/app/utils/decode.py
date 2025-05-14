import tensorflow as tf
from keras.layers import StringLookup
import numpy as np
import keras
from app.utils.config import max_len


# Load the vocabulary from vocab.txt
with open("app/models/vocab.txt", "r") as f:
    characters = [line.strip() for line in f.readlines()]

# Create StringLookup layers
num_to_char = StringLookup(vocabulary=characters, mask_token=None, invert=True)

def decode_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.ops.nn.ctc_decode(pred, sequence_lengths=input_len)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = (
            tf.strings.reduce_join(num_to_char(res))
            .numpy()
            .decode("utf-8")
            .replace("[UNK]", "")
        )
        output_text.append(res)
    return output_text
