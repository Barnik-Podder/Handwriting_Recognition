from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)

    CORS(app)

    @app.route("/")
    def health_check():
        return "Server is running!"

    from .routes.predict import predict_bp
    app.register_blueprint(predict_bp)

    return app
