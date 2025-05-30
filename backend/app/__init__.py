from flask import Flask, request, jsonify
from flask_cors import CORS
import psutil
import os
import tensorflow as tf
from app.services.model_handler import load_handwriting_model, predict_sentence


def log_memory_usage(context=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # in MB
    print(f"[Memory Usage] {context}: {mem:.2f} MB")


def create_app():
    app = Flask(__name__)
    CORS(app)

    log_memory_usage("Before loading model")
    model = load_handwriting_model()
    log_memory_usage("After loading model")

    @app.route("/")
    def health_check():
        return "Server is running!"

    @app.route("/predict", methods=["POST"])
    def predict():
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        log_memory_usage("Before prediction")
        sentence = predict_sentence(file, model)
        log_memory_usage("After prediction")

        return jsonify({"prediction": sentence})
    
    return app
