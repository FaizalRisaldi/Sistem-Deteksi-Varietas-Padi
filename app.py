from flask import Flask, render_template, request
import os
from utils.model_loader import load_model
from utils.inference import run_inference

app = Flask(__name__)

model = load_model("model/best_model.pth", num_classes=4)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

# ✅ AUTO CREATE FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route("/")
def dashboard():
    return render_template("index.html", title="Dashboard")


@app.route("/deteksi")
def deteksi():
    return render_template("deteksi.html", title="Deteksi")


@app.route("/panduan")
def panduan():
    return render_template("panduan.html", title="Panduan")


@app.route("/hasil", methods=["POST"])
def hasil():
    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    filename = file.filename

    # ✅ 1. buat path
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # ✅ 2. simpan file
    file.save(filepath)

    # ✅ 3. path untuk ditampilkan (original)
    original_path = f"uploads/{filename}"

    # ✅ 4. jalankan model
    result_path, total, ciherang, ir64, mentik, boxes = run_inference(
        model, filepath, RESULT_FOLDER
    )

    # ✅ 5. rapikan path hasil
    result_path = result_path.replace("\\", "/")
    result_path = result_path.split("static/")[-1]

    return render_template(
        "hasil.html",
        original_path=original_path,
        result_path=result_path,
        total=total,
        ciherang=ciherang,
        ir64=ir64,
        mentik=mentik,
        boxes=boxes
    )


if __name__ == "__main__":
    app.run(debug=True)