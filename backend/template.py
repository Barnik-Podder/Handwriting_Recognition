import os

BASE_DIR = "backend"

folders = [
    "app",
    "app/routes",
    "app/services",
    "app/utils",
    "app/models",
    "uploads"
]

files = {
    "run.py": '''from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
''',

    "requirements.txt": "flask\ntensorflow\nopencv-python\nnumpy\nPillow\n",

    ".gitignore": "*.pyc\n__pycache__/\n.env\nuploads/\n",

    "README.md": "# Handwriting Recognition Backend\n\nThis backend receives a handwritten sentence image, segments it into words, uses a trained model to recognize each word, and returns the full sentence as JSON.",

    "app/__init__.py": '''from flask import Flask

def create_app():
    app = Flask(__name__)

    from .routes.predict import predict_bp
    app.register_blueprint(predict_bp, url_prefix="/api")

    return app
''',

    "app/routes/__init__.py": "",

    "app/routes/predict.py": '''from flask import Blueprint, request, jsonify
from app.services.model_handler import load_model, predict_sentence

predict_bp = Blueprint("predict", __name__)
model = load_model()

@predict_bp.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    sentence = predict_sentence(file, model)
    return jsonify({"prediction": sentence})
''',

    "app/services/__init__.py": "",

    "app/services/model_handler.py": '''import cv2
import numpy as np
from tensorflow.keras.models import load_model
from app.utils.segment import segment_into_words
from app.utils.preprocess import preprocess_image
from app.utils.decode import decode_predictions

def load_model():
    return load_model("app/models/handwriting_model")

def predict_sentence(image_file, model):
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    word_imgs = segment_into_words(img)
    words = []

    for word_img in word_imgs:
        processed = preprocess_image(word_img)
        pred = model.predict(np.expand_dims(processed, axis=0))
        word = decode_predictions(pred)
        words.append(word)

    return " ".join(words)
''',

    "app/utils/__init__.py": "",

    "app/utils/segment.py": '''import cv2
import numpy as np

def segment_into_words(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_images = []

    for cnt in sorted(contours, key=lambda x: cv2.boundingRect(x)[0]):
        x, y, w, h = cv2.boundingRect(cnt)
        word = image[y:y+h, x:x+w]
        word_images.append(word)

    return word_images
''',

    "app/utils/preprocess.py": '''import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(img, size=(128, 32)):
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img
''',

    "app/utils/decode.py": '''import tensorflow as tf
from keras.layers import StringLookup

# You need to initialize `num_to_char` using your training vocabulary
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
num_to_char = StringLookup(vocabulary=list(characters), mask_token=None, invert=True)

def decode_predictions(pred):
    input_len = tf.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        text = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(text.replace("[UNK]", ""))
    return output_text[0]  # Return first decoded word
'''
}

def create_project_structure():
    os.makedirs(BASE_DIR, exist_ok=True)

    for folder in folders:
        os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

    for path, content in files.items():
        full_path = os.path.join(BASE_DIR, path)
        folder = os.path.dirname(full_path)
        os.makedirs(folder, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    print(f"✅ Folder structure for '{BASE_DIR}' created successfully!")

if __name__ == "__main__":
    create_project_structure()
