# Human Fall Detection Using YOLO11

## Project Overview
This project explores human fall detection using the YOLO11 object detection model. The goal is to identify fallen persons in images using bounding boxes and confidence scores.

## Objectives
- Detect fallen persons in images.
- Draw bounding boxes around detected persons.
- Display model confidence scores.
- Evaluate the trained model on a separate test dataset.

## Technologies Used
- Python
- YOLO11
- Ultralytics
- OpenCV
- Matplotlib
- Google Colab

## Project Workflow
1. Load and inspect the annotated dataset.
2. Configure the dataset using `data.yaml`.
3. Train the YOLO11 model.
4. Load the trained model (`best.pt`).
5. Run predictions on test images.
6. Visualize detected objects with bounding boxes.
7. Evaluate model performance.

## Project Structure
```text
Human-Fall-Detection-YOLO11/
├── README.md
├── requirements.txt
├── data.yaml
├── train.py
├── predict.py
├── evaluate.py
└── results/
```

## Installation
```bash
pip install -r requirements.txt
```

## Prediction
```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(
    source="test_image.jpg",
    conf=0.25,
    save=True
)
```

## Model Evaluation
Evaluation metrics such as Precision, Recall, mAP50 and mAP50-95 will be reported after testing.

## Future Improvements
- Improve detection accuracy using a more representative dataset.
- Test on videos and real-time camera streams.
- Explore deployment for practical monitoring applications.

## Note
This is an experimental computer vision project. Detection performance depends on dataset quality, annotations and testing conditions.
