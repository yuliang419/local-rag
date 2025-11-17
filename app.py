# pylint: disable=wrong-import-position
# flake8: noqa: E402

"""Defines routes for embedding files to the vector database, and retrieving the response from the model."""

import os
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, jsonify
from embed import embed
from query import query

TEMP_FOLDER = os.getenv("TEMP_FOLDER", "./_temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)

app = Flask(__name__)


@app.route("/embed", methods=["POST"])
def route_embed():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    embedded = embed(file)

    if embedded:
        return jsonify({"message": "File embedded successfully"}), 200

    return jsonify({"error": "File embedding failed"}), 400


@app.route("/query", methods=["POST"])
def route_query():
    data = request.get_json()
    response = query(data.get("query"))

    if response:
        return jsonify({"message": response}), 200

    return jsonify({"error": "Route query failed"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
