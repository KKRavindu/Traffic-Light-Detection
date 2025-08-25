from ultralytics import YOLO
model = YOLO("YOLO11m.pt")  # load a pretrained model (recommended for training)

model.train(data= "dataset_custom.yaml",imgsz=640
            ,epochs=100,workers=0,device="cpu")
