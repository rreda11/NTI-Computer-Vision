
from pyexpat import model

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.predict(source=0, show=True)

# Run on a video
model.predict(source='video.mp4', save=True)