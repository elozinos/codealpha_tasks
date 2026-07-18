from ultralytics import YOLO
model=YOLO("best3.pt")
model.predict(source="img_1.png", save=True, conf=0.10)
model.export(format="onnx")
