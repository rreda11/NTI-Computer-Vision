# Vehicle Detection with YOLO

This project uses the **Ultralytics YOLO model** to detect vehicles in images.  
The trained model is able to identify **three classes**:  

- **Car**  
- **Truck**  
- **Bus**

---

## Dataset

The dataset was annotated in YOLO format.  
It is organized as follows:

dataset/
├── train/ # Training images
├── valid/ # Validation images
└── test/ # Test images

The `data.yaml` file defines the dataset paths and class names.

---

## Model Training

We used YOLO11n (a small version of YOLO) as the base model.

Training configuration:
- **Epochs:** 10  
- **Batch size:** 3  
- **Confidence threshold:** 0.25  

Training command:
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="data.yaml", epochs=10, batch=3, conf=0.25)