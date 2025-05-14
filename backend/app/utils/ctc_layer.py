import tensorflow as tf
from tensorflow.keras.layers import Layer

class CTCLayer(Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute shapes
        batch_size = tf.shape(y_true)[0]
        input_length = tf.shape(y_pred)[1]
        label_length = tf.shape(y_true)[1]

        # Convert to int64 to match model dtype expectations
        input_length = tf.ones((batch_size, 1), dtype=tf.int64) * tf.cast(input_length, tf.int64)
        label_length = tf.ones((batch_size, 1), dtype=tf.int64) * tf.cast(label_length, tf.int64)

        # Compute the CTC loss
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # Return y_pred for inference
        return y_pred
