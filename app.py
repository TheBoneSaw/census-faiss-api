from flask import Flask, request, jsonify
import faiss
import json
import numpy as np
import requests

app = Flask(__name__)

# === Config ===
INDEX_PATH = "census_index.faiss"
CHUNK_MAP_FILE = "chunk_map.json"
NETLIFY_BASE = "https://67f29c41c2817a5c8a60cef7--polite-sunshine-a568af.netlify.app/chunks_by_size"

# === Load FAISS and Chunk Map ===
index = faiss.read_index(INDEX_PATH)

with open(CHUNK_MAP_FILE, "r") as f:
    chunk_map = json.load(f)

def fetch_chunk(chunk_file):
    url = f"{NETLIFY_BASE}/{chunk_file}"
    resp = requests.get(url)
    return resp.json()

@app.route("/search", methods=["POST"])
def search():
    body = request.get_json()
    embedding = body.get("embedding")
    limit = int(body.get("limit", 5))

    if not embedding or not isinstance(embedding, list):
        return jsonify({"error": "Missing or invalid embedding"}), 400

    query_vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query_vector, limit)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        chunk_file = chunk_map.get(str(idx))
        if not chunk_file:
            continue
        chunk = fetch_chunk(chunk_file)
        entry = next((e for e in chunk if e["global_id"] == idx), None)
        if entry:
            results.append(entry)

    return jsonify(results)
