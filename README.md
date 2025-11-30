# 🚗 Real-Time Driver Monitoring System (DMS)

A real-time Driver Monitoring System that detects **drowsiness** and **distracted behaviors** using deep learning, applies temporal rules to avoid false alarms, and provides **visual and audio alerts** to improve driving safety.

---

## 📌 Project Overview
This project was developed as part of the DEPI training program.  
It aims to:
- Detect **driver drowsiness** using CNN/MobileNetV2.
- Detect **distraction behaviors** (phone use, smoking, eating, etc.).
- Apply **temporal rules** to avoid missclassifying blinking with drowsiness.
- Provide **visual & sound alerts** in real-time on webcam feed.
- Log all events (drowsiness, distractions) (Optinoal feature).

---

## 🖼️ Final Interface
- **Live Webcam Feed** – Shows real-time detection.
- **Status Panel** – Displays current driver state (Drowsy, Alert, Distracted).
- **Alerts** – Visual (red border/text) + Audio (buzzer/beep)(optional).
- **Logging System** – Saves `.csv` file after each session with(optional):
  - Timestamp of events
  - Event type (Drowsy/Distraction)
  - Duration of event
  - Alert triggered (Yes/No)

---
## ⚙️ Tech Stack

- **Python** (TensorFlow / OpenCV / NumPy / Pandas / Pytorch)
- **Yolo** for detection and classification
- **Matplotlib/ Seaborn** for visualizations
- **PyAudio / Playsound** for audio alerts (Optional)
