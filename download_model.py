import os
import gdown

MODEL_PATH = "model/best_model.pth"

FILE_ID = "1ltjzBxA6hjzKDXPLt-NRWlwF2w45l6tT"

def download_model():
    if os.path.exists(MODEL_PATH):
        print("Model already exists")
        return

    os.makedirs("model", exist_ok=True)

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print("Downloading model...")
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Download selesai")
