from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import base64
import numpy as np
import cv2
import mediapipe as mp
import math
import time

app = Flask(__name__)
CORS(app)

cuttoff = 0.5
high_cuttoff = 0.8
conf_threshold = 0.45
iou_threshold = 0.5

# ---------------------------------------------------
# LOAD MODELS (Distraction Only - ONNX Runtime)
# ---------------------------------------------------

# Load the ONNX model
model_path = "Ultimate4_augmentations_70epochs.onnx"
providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if ort.get_device() == "GPU"
    else ["CPUExecutionProvider"]
)
distraction_session = ort.InferenceSession(model_path, providers=providers)

# Get model input details
model_inputs = distraction_session.get_inputs()
input_shape = model_inputs[0].shape
input_height = input_shape[2]
input_width = input_shape[3]
input_name = model_inputs[0].name

label_lup = {0: "Drinking", 1: "Person", 2: "Smoking", 3: "Using Phone"}
respone_lup = {
    "Drinking": "drinking",
    "Person": "default",
    "Smoking": "smoking",
    "Using Phone": "phone",
}

print("\nDistraction model loaded via ONNX Runtime.")


# ---------------------------------------------------
# YOLO ONNX HELPER FUNCTIONS
# ---------------------------------------------------
def preprocess_image(img):
    """
    Resizes and pads image to match model input shape (Letterboxing),
    normalizes to 0-1, and converts to NCHW format.
    """
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    # Calculate scale
    scale = min(input_width / w, input_height / h)
    nw, nh = int(w * scale), int(h * scale)

    # Resize
    image_resized = cv2.resize(image_rgb, (nw, nh))

    # Create padded image
    image_padded = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    dw = (input_width - nw) // 2
    dh = (input_height - nh) // 2
    image_padded[dh : nh + dh, dw : nw + dw, :] = image_resized

    # Normalize and Transpose
    input_tensor = image_padded / 255.0
    input_tensor = input_tensor.transpose(2, 0, 1)  # HWC to CHW
    input_tensor = input_tensor[np.newaxis, :, :, :].astype(np.float32)

    return input_tensor, scale, dw, dh


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right"""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def postprocess_output(outputs, scale, dw, dh):
    """
    Parses YOLOv8 ONNX output.
    Output shape is usually [1, 4 + num_classes, 8400].
    """
    # Outputs is a list, we need the first element
    # Shape: (1, 84, 8400) -> 4 box coords + 80 classes (or custom num classes)
    predictions = np.squeeze(outputs[0]).T  # Transpose to (8400, 4+nc)

    # Filter out low confidence scores
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        return [], []

    # Get class IDs
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Extract boxes (cx, cy, w, h)
    boxes = predictions[:, :4]

    # Convert boxes to [x1, y1, x2, y2]
    boxes = xywh2xyxy(boxes)

    # Scale boxes back to original image dimensions (remove padding)
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, 2] -= dw
    boxes[:, 3] -= dh
    boxes /= scale

    # Apply NMS (Non-Maximum Suppression)
    # cv2.dnn.NMSBoxes expects boxes as [x, y, w, h], not xyxy
    # So we convert xyxy back to xywh just for NMS
    nms_boxes = boxes.copy()
    nms_boxes[:, 2] = nms_boxes[:, 2] - nms_boxes[:, 0]  # width
    nms_boxes[:, 3] = nms_boxes[:, 3] - nms_boxes[:, 1]  # height

    indices = cv2.dnn.NMSBoxes(
        nms_boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold
    )

    final_labels = []
    final_confs = []

    if len(indices) > 0:
        for i in indices.flatten():
            final_labels.append(class_ids[i])
            final_confs.append(scores[i])

    return final_labels, final_confs


# ---------------------------------------------------
# DROWSINESS DETECTOR (MediaPipe + State)
# ---------------------------------------------------
class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Landmarks
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.MOUTH_LEFT = 78
        self.MOUTH_RIGHT = 308

        # State Timers
        self.eyes_closed_start = None
        self.yawn_start = None

        # Thresholds (Matches drowsiness_alarm.py)
        self.EAR_THRESH = 0.20
        self.EYE_CLOSED_SECS = 1.0

        self.MOUTH_THRESH = 0.30
        self.YAWN_SECS = 0.8

    def _euclidean(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _eye_aspect_ratio(self, landmarks, indices):
        p1 = landmarks[indices[0]]
        p2 = landmarks[indices[1]]
        p3 = landmarks[indices[2]]
        p4 = landmarks[indices[3]]
        p5 = landmarks[indices[4]]
        p6 = landmarks[indices[5]]
        A = self._euclidean(p2, p6)
        B = self._euclidean(p3, p5)
        C = self._euclidean(p1, p4)
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)

    def process_frame(self, img):
        """
        Returns 'high', 'medium', or 'No' based on timers.
        """
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        drowsiness_status = "No"

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            pts = [(p.x * w, p.y * h) for p in lm]

            # 1. Calc EAR
            left_ear = self._eye_aspect_ratio(pts, self.LEFT_EYE)
            right_ear = self._eye_aspect_ratio(pts, self.RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            # 2. Calc Mouth Ratio
            top = pts[self.MOUTH_TOP]
            bottom = pts[self.MOUTH_BOTTOM]
            left = pts[self.MOUTH_LEFT]
            right = pts[self.MOUTH_RIGHT]
            mouth_v = self._euclidean(top, bottom)
            mouth_h = self._euclidean(left, right)
            mouth_ratio = mouth_v / (mouth_h + 1e-6)

            current_time = time.monotonic()

            # --- Logic: Eyes ---
            if ear < self.EAR_THRESH:
                if self.eyes_closed_start is None:
                    self.eyes_closed_start = current_time
                else:
                    elapsed = current_time - self.eyes_closed_start
                    if elapsed >= self.EYE_CLOSED_SECS:
                        drowsiness_status = "high"
                    elif elapsed >= (self.EYE_CLOSED_SECS * 0.5):
                        # Mark as medium if halfway to threshold
                        if drowsiness_status != "high":
                            drowsiness_status = "medium"
            else:
                self.eyes_closed_start = None

            # --- Logic: Yawn ---
            if mouth_ratio > self.MOUTH_THRESH:
                if self.yawn_start is None:
                    self.yawn_start = current_time
                else:
                    elapsed = current_time - self.yawn_start
                    if elapsed >= self.YAWN_SECS:
                        # If already high from eyes, keep high, else high from yawn
                        drowsiness_status = "high"
            else:
                self.yawn_start = None
        else:
            # No face detected, reset timers
            self.eyes_closed_start = None
            self.yawn_start = None

        return drowsiness_status


# Initialize detector globally
drowsiness_detector = DrowsinessDetector()
print("MediaPipe Drowsiness Detector initialized.")


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
    # 1) RUN DISTRACTION MODEL (ONNX)
    # -----------------------------------------------

    # Preprocess
    input_tensor, scale, dw, dh = preprocess_image(img)

    # Inference
    outputs = distraction_session.run(
        [distraction_session.get_outputs()[0].name], {input_name: input_tensor}
    )

    # Postprocess
    detected_labels, detected_confs = postprocess_output(outputs, scale, dw, dh)

    final_results = {
        "drowsiness": "No",
        "drinking": "No",
        "phone": "No",
        "smoking": "No",
    }

    # Process Distractions
    for label, conf in zip(detected_labels, detected_confs):
        name = label_lup.get(label, "Person")
        if name != "Person":
            if conf > high_cuttoff:
                final_results[respone_lup[name]] = "high"
            elif conf > cuttoff:
                final_results[respone_lup[name]] = "medium"

    # -----------------------------------------------
    # 2) RUN DROWSINESS DETECTOR (MediaPipe)
    # -----------------------------------------------
    # This updates internal state timers based on the current frame
    drowsiness_state = drowsiness_detector.process_frame(img)
    final_results["drowsiness"] = drowsiness_state

    return jsonify(final_results)


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
