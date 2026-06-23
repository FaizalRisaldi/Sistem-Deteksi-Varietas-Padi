import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
import os

# mapping class
CLASS_NAMES = ["background", "Ciherang", "IR64", "Mentik"]

transform = T.Compose([
    T.ToTensor()
])

def run_inference(model, image_path, output_dir, score_threshold=0.5):
    model.eval()

    # ✅ Load image dari path
    image = Image.open(image_path).convert("RGB")

    # transform ke tensor
    img_tensor = transform(image)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]

    results = []

    total = 0
    ciherang = 0
    ir64 = 0
    mentik = 0

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    W, H = image.size  # ⬅️ penting buat normalisasi

    for box, score, label in zip(boxes, scores, labels):
        if score >= score_threshold:
            x1, y1, x2, y2 = box.tolist()

            label_name = CLASS_NAMES[label.item()]

            # ✅ hitung total & per kelas
            total += 1
            if label_name == "Ciherang":
                ciherang += 1
            elif label_name == "IR64":
                ir64 += 1
            elif label_name == "Mentik":
                mentik += 1

            # ✅ simpan untuk frontend (RELATIVE COORDINATE)
            results.append({
                "label": label_name,
                "conf": float(score.item()),
                "x": x1 / W,
                "y": y1 / H,
                "w": (x2 - x1) / W,
                "h": (y2 - y1) / H,
            })

            COLORS = {
                "Ciherang": "red",
                "IR64": "yellow",
                "Mentik": "lime"
            }

            class_name = CLASS_NAMES[label]
            color = COLORS.get(class_name, "white")

            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2,
                edgecolor=color
            )
            ax.add_patch(rect)

            ax.text(
                x1,
                y1 - 5,
                f"{class_name} {score:.2f}",
                color=color,
                fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5, pad=1)
            )

    plt.axis('off')

    # ✅ simpan hasil gambar
    filename = os.path.basename(image_path)
    result_path = os.path.join(output_dir, filename)

    plt.savefig(result_path, bbox_inches='tight', dpi=300)
    plt.close()

    result_path = result_path.replace("\\", "/")

    return result_path, total, ciherang, ir64, mentik, results