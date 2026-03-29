import os
import uuid
from flask import Flask, request, jsonify, send_file, render_template
from ultralytics import YOLO
from processor import process_video

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Loading model...")
model = YOLO("plate_detector.pt").to("mps")
print("Model loaded")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():

    if "video" not in request.files:
        return jsonify({"error":"no file"}),400

    file = request.files["video"]

    file_id = str(uuid.uuid4())

    input_path = os.path.join(UPLOAD_FOLDER, file_id + ".mp4")
    output_path = os.path.join(OUTPUT_FOLDER, file_id + "_blurred.mp4")

    file.save(input_path)

    process_video(input_path, output_path, model)

    return jsonify({
        "download_url": f"/download/{file_id}"
    })


@app.route("/download/<file_id>")
def download_video(file_id):

    path = os.path.join(
        OUTPUT_FOLDER,
        file_id + "_blurred.mp4"
    )

    if not os.path.exists(path):
        return jsonify({"error":"file not ready"}),404

    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )