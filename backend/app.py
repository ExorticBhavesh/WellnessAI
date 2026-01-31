#!/usr/bin/env python3
"""
Health Chatbot System – Flask Backend
Supports:
- REST API for frontend
- Optional CLI mode (--cli)
"""

import os
import argparse
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database.database import engine, Base
# --------------------------------------------------
# Flask App Setup (MUST BE FIRST)
# --------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --------------------------------------------------
# SAFE IMPORTS (AFTER app is created)
# --------------------------------------------------
from chatbot.bot_logic import HealthChatbot

chatbot = HealthChatbot()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "health_dataset.csv")
AUDIO_CACHE = os.path.join(BASE_DIR, "audio", "cache")
os.makedirs(AUDIO_CACHE, exist_ok=True)

# --------------------------------------------------
# HOME
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Health Chatbot Backend Running",
        "version": "1.0.0",
        "endpoints": [
            "/chat [POST]",
            "/predict [POST]",
            "/tts [POST]",
            "/dataset [GET]",
            "/train [POST]",
            "/health [GET]"
        ]
    })

# --------------------------------------------------
# CHAT API
# --------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json(silent=True)

    if not data or "message" not in data:
        return jsonify({"error": "Message is required"}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    reply = chatbot.process_message(message)
    return jsonify({"reply": reply})

# --------------------------------------------------
# PREDICTION API
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        from model.predict import predict_single
        data = request.get_json()
        return jsonify(predict_single(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# TEXT TO SPEECH API
# --------------------------------------------------
@app.route("/tts", methods=["POST"])
def tts_api():
    try:
        from audio.tts import TextToSpeechService
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Text required"}), 400

        tts = TextToSpeechService()
        audio_path = tts.synthesize_speech(text, save_to_file=True)

        return jsonify({
            "success": True,
            "audio_path": audio_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# DATASET INFO
# --------------------------------------------------
@app.route("/dataset", methods=["GET"])
def dataset_info():
    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": "Dataset not found"}), 404

    df = pd.read_csv(DATASET_PATH)
    return jsonify({
        "rows": len(df),
        "columns": list(df.columns),
        "sample": df.head(5).to_dict("records")
    })

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
@app.route("/train", methods=["POST"])
def train_model():
    try:
        from model.train_models import train_health_models
        train_health_models()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "flask": True,
        "chatbot": True,
        "dataset": os.path.exists(DATASET_PATH)
    })

# --------------------------------------------------
# CLI MODE
# --------------------------------------------------
def run_cli():
    print("\n🏥 HEALTH CHATBOT CLI\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in ("exit", "quit"):
            break
        print("Bot:", chatbot.process_message(msg))

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        print(f"\n🚀 Backend running at http://{args.host}:{args.port}\n")
        app.run(host=args.host, port=args.port, debug=args.debug)
