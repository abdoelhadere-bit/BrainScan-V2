from ultralytics import YOLO

model = YOLO("models/final/best.pt")

print("Model loaded successfully!")
print("Task:", model.task)
print("Classes:", model.names)
