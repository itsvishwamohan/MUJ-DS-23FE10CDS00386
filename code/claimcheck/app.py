"""Flask web UI.  python app.py  ->  http://127.0.0.1:5000"""
import threading

from flask import Flask, jsonify, render_template, request

from config import ROOT, load_config
from llm_client import OllamaError
from pipeline import ClaimCheckPipeline

app = Flask(__name__)
_lock = threading.RLock()  # one local model: process one request at a time
_pipeline = None


def get_pipeline() -> ClaimCheckPipeline:
    global _pipeline
    with _lock:
        if _pipeline is None:
            _pipeline = ClaimCheckPipeline.from_config()
        return _pipeline


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/samples")
def samples():
    folder = ROOT / "samples"
    return jsonify([{"name": p.stem.replace("_", " "), "text": p.read_text(encoding="utf-8")}
                    for p in sorted(folder.glob("*.txt"))])


@app.get("/api/health")
def health():
    pipe = get_pipeline()
    try:
        pipe.client.check_ready()
        return jsonify({"ok": True, "model": pipe.client.model})
    except OllamaError as exc:
        return jsonify({"ok": False, "model": pipe.client.model, "error": str(exc)})


@app.post("/api/check")
def check():
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
    limit = load_config()["app"]["max_input_chars"]
    if not text:
        return jsonify({"error": "Paste some text to check."}), 400
    if len(text) > limit:
        return jsonify({"error": f"Text is too long ({len(text)} characters; the limit is {limit})."}), 400
    try:
        with _lock:
            pipe = get_pipeline()
            pipe.client.check_ready()
            return jsonify(pipe.run(text))
    except OllamaError as exc:
        return jsonify({"error": str(exc)}), 502


if __name__ == "__main__":
    cfg = load_config()["app"]
    app.run(host=cfg["host"], port=cfg["port"], debug=False)
