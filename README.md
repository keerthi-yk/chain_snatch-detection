# 🔗 Chain Snatch Detection System

> An AI-powered surveillance analytics pipeline that detects chain-snatching incidents in real time using computer vision, pose estimation, motion analysis, and multi-factor risk scoring.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Modules](#modules)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Results & Outputs](#results--outputs)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

Chain snatching is one of the most prevalent street crimes in urban India, particularly targeting pedestrians and motorcyclists. This project presents a **multi-module computer vision and machine learning pipeline** designed to automatically detect potential chain-snatching events from CCTV footage.

The system combines **human pose estimation**, **vehicle detection**, **motion analysis**, **gesture recognition**, **trajectory modelling**, and **risk scoring** into a unified detection framework that can flag suspicious activity in real time.

---

## Problem Statement

Traditional CCTV surveillance requires constant human monitoring, which is impractical at scale. This system automates the detection of chain-snatching behavior by analyzing:

- Suspicious proximity between individuals and two-wheelers
- Abrupt snatch-like gestures using body keypoint analysis
- Sudden acceleration and escape patterns post-incident
- Trajectory anomalies and loitering behavior
- Per-person risk scoring based on aggregated behavioral signals

---

## System Architecture

```
CCTV Video Input
        │
        ▼
┌─────────────────────┐
│  Object Detection   │  ← YOLO-based person & vehicle detection
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Pose Estimation    │  ← MediaPipe / OpenPose keypoint extraction
└────────┬────────────┘
         │
    ┌────┴────────────────────────────┐
    │                                 │
    ▼                                 ▼
┌──────────────────┐      ┌──────────────────────────┐
│ Gesture          │      │  Motion Pattern Analysis  │
│ Recognition      │      │  & Trajectory Modelling   │
└────────┬─────────┘      └────────────┬─────────────┘
         │                             │
         └──────────┬──────────────────┘
                    ▼
         ┌─────────────────────┐
         │  Car/Vehicle Model  │
         │  Detection          │
         └────────┬────────────┘
                  │
                  ▼
         ┌─────────────────────┐
         │  Escape Acceleration│
         │  Detection          │
         └────────┬────────────┘
                  │
                  ▼
         ┌─────────────────────┐
         │  Risk Score per     │  ← Composite scoring engine
         │  Person             │
         └────────┬────────────┘
                  │
                  ▼
         🚨 Alert / Flagged Incident Output
```

---

## Modules

### 1. `chain-snaching-project.ipynb` — Core Pipeline
The master notebook that integrates all sub-modules. Handles video ingestion, frame-by-frame processing, and coordinates detection outputs across the entire system.

### 2. `snatching-gesture-recognition-mine.ipynb` — Gesture Recognition
Analyzes human body keypoints to detect rapid, arm-extension gestures associated with chain snatching. Uses pose landmark trajectories to identify suspicious reach-and-pull movements.

### 3. `motion-pattern-analysis-task.ipynb` — Motion Pattern Analysis
Tracks and analyzes the movement patterns of detected persons across frames. Identifies loitering, sudden directional changes, and proximity events near potential victims.

### 4. `trajectory-modelling.ipynb` — Trajectory Modelling
Models the spatial paths of individuals and vehicles over time. Flags trajectories that follow known pre- and post-snatch behavioral patterns such as slow circling followed by rapid escape.

### 5. `car-model-detection.ipynb` — Vehicle Detection
Detects and classifies two-wheelers (motorcycles/scooters) — the primary vehicles used in chain-snatching incidents. Uses object detection to log vehicle presence, position, and speed.

### 6. `escape-acceleration-detection.ipynb` — Escape & Acceleration Detection
Identifies high-acceleration escape events in the immediate aftermath of a potential snatching. Tracks sudden speed changes in vehicle or person trajectories as a post-event signal.

### 7. `risk-score-per-person-task.ipynb` — Risk Scoring Engine
Aggregates signals from all other modules to compute a per-person composite risk score. Individuals exceeding a defined threshold are flagged as high-risk and trigger an alert.

---

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.8+ |
| Notebooks | Jupyter Notebook |
| Computer Vision | OpenCV |
| Object Detection | YOLOv5 / YOLOv8 |
| Pose Estimation | MediaPipe / OpenPose |
| Deep Learning | PyTorch / TensorFlow |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Tracking | DeepSORT / ByteTrack |

---

## Project Structure

```
chain_snatch-detection/
│
├── chain-snaching-project.ipynb          # Core integration pipeline
├── snatching-gesture-recognition-mine.ipynb  # Gesture recognition module
├── motion-pattern-analysis-task.ipynb    # Motion analysis module
├── trajectory-modelling.ipynb            # Trajectory modelling module
├── car-model-detection.ipynb             # Vehicle detection module
├── escape-acceleration-detection.ipynb   # Escape detection module
├── risk-score-per-person-task.ipynb      # Risk scoring engine
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- A CUDA-capable GPU (recommended for real-time inference)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/keerthi-yk/chain_snatch-detection.git
cd chain_snatch-detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install opencv-python torch torchvision mediapipe numpy pandas matplotlib seaborn jupyter

# 4. Launch Jupyter
jupyter notebook
```

### Running the Pipeline

Open the notebooks in the following recommended order:

1. `car-model-detection.ipynb` — Set up vehicle detection
2. `snatching-gesture-recognition-mine.ipynb` — Configure gesture recognition
3. `motion-pattern-analysis-task.ipynb` — Run motion analysis
4. `trajectory-modelling.ipynb` — Build trajectory models
5. `escape-acceleration-detection.ipynb` — Configure escape detection
6. `risk-score-per-person-task.ipynb` — Calibrate risk scoring
7. `chain-snaching-project.ipynb` — Run the full integrated pipeline

---

## How It Works

The system processes CCTV video frames through a sequential detection pipeline:

**Step 1 — Detection:** YOLO detects all persons and two-wheelers in each frame.

**Step 2 — Tracking:** DeepSORT/ByteTrack assigns consistent IDs to detected individuals across frames, enabling longitudinal behavioral analysis.

**Step 3 — Pose Estimation:** MediaPipe extracts 33 body keypoints per person per frame, feeding into gesture recognition.

**Step 4 — Behavioral Analysis:** Motion patterns, trajectories, and gestures are analyzed simultaneously. Key signals include arm velocity spikes (snatch gesture), loitering near victims, and sudden directional reversals.

**Step 5 — Vehicle Correlation:** Two-wheeler proximity and speed changes are correlated with pedestrian behavioral events to identify co-occurring incident signals.

**Step 6 — Risk Scoring:** Each tracked person receives a composite risk score based on weighted signals from all modules. High-risk individuals (score above threshold) trigger an alert.

**Step 7 — Alert Generation:** Flagged frames are annotated and logged with timestamps, bounding boxes, keypoints, and the contributing risk factors.

---

## Results & Outputs

- Annotated video frames with bounding boxes, pose overlays, and risk labels
- Per-person risk score timeline plots
- Incident event logs with frame number, timestamp, and detected signals
- Escape trajectory visualizations
- Gesture confidence heatmaps

---

## Future Work

- [ ] Real-time streaming pipeline integration (RTSP / IP camera feeds)
- [ ] Web-based dashboard for live monitoring and alert management
- [ ] Fine-tuned model on India-specific CCTV datasets
- [ ] Integration with police alert systems via API
- [ ] Multi-camera scene stitching for wide-area coverage
- [ ] Edge deployment on NVIDIA Jetson / Raspberry Pi

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

---

## License

This project is intended for academic and research purposes. Please use responsibly and in compliance with applicable privacy laws and regulations.

---

*Built with a focus on public safety and smart city surveillance solutions.*
