from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import base64
import numpy as np
import cv2
import os

app = Flask(__name__)
CORS(app)

cuttoff = 0.3
high_cuttoff = 0.5

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

# Model 1 model (phone, drink, smoke)
distraction_model = YOLO("Ultimate4_augmentations_70epochs.pt")

label_lup = {0: "Drinking", 1: "Person", 2: "Smoking", 3: "Using_Phone"}
respone_lup = {
    "Drinking": "drinking",
    "Person": "default",
    "Smoking": "smoking",
    "Using_Phone": "phone",
}

# Model 2 drowsiness model (if not ready, skip)
drowsiness_model = (
    YOLO("drowsiness_classifier.pt")
    if os.path.exists("drowsiness_classifier.pt")
    else None
)

print("\nDistraction model loaded.")
if drowsiness_model:
    print("Drowsiness model loaded.")
else:
    print("Drowsiness model NOT found. Using default 'No' values.")


# ---------------------------------------------------
# DECODE BASE64 IMAGE
# ---------------------------------------------------
def decode_base64(data_url):
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# ---------------------------------------------------
# ENDPOINT: /infer
# ---------------------------------------------------
@app.post("/infer")
def infer():
    data = request.get_json()
    img = decode_base64(data["frame"])

    # -----------------------------------------------
    # 1) RUN DISTRACTION MODEL
    # -----------------------------------------------
    results_ls = distraction_model(img, imgsz=640, conf=0.45)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # for r in results_ls:
    #     r = r.cpu()
    #     for label, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
    #         print(f"detected: {label_lup[int(label)]} with conf {conf}")
    #     print(">>>>>>>>>>>>>>")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    behavior_results = results_ls[0].cpu()

    final_results = {
        "drowsiness": "No",
        "drinking": "No",
        "phone": "No",
        "smoking": "No",
    }

    for label, conf in zip(
        behavior_results.boxes.cls.tolist(), behavior_results.boxes.conf.tolist()
    ):
        name = label_lup[label]
        if name != "Person":
            if conf > high_cuttoff:
                final_results[respone_lup[name]] = "high"
            elif conf > cuttoff:
                final_results[respone_lup[name]] = "medium"

    if drowsiness_model:
        d_results = drowsiness_model(img, imgsz=640)[0].cpu()
        if d_results.probs.top1 == 0:
            if d_results.probs.top1conf.item() > 0.9:
                final_results["drowsiness"] = "high"
            elif d_results.probs.top1conf.item() > 0.8:
                final_results["drowsiness"] = "medium"

    response = final_results
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(response)
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(jsonify(response))
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return jsonify(response)


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
