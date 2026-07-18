from ultralytics import YOLO
model= YOLO("yolo26m.pt")
model.train(data="dataset_custom.yaml", epochs=10, imgsz=640, batch=8, device="cpu", workers=1)