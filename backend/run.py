# run.py
import os
import warnings

# --- Suppress TensorFlow and Deprecation Warnings ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all, 1 = info, 2 = warning, 3 = error
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # disable oneDNN logs
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
