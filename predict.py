from ultralytics import YOLO
model=YOLO("yolov11m_custom.pt")
model.predict(source="traffic_light_3.mp4",show=True,save=True,conf=0.5,line_width=2)