from flask import Blueprint, request, jsonify
from app.services.model_handler import load_handwriting_model, predict_sentence

predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    model = load_handwriting_model()
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    sentence = predict_sentence(file, model)
    return jsonify({"prediction": sentence})
