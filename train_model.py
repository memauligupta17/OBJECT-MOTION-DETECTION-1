"""
Custom YOLOv8 model training script.

Apne specific objects detect karne ke liye custom dataset par
fine-tune karo.

Dataset structure (YOLO format):
    dataset/
        images/train/  *.jpg
        images/val/    *.jpg
        labels/train/  *.txt
        labels/val/    *.txt
        data.yaml

data.yaml example:
    path: ./dataset
    train: images/train
    val: images/val
    names:
        0: person
        1: car

Labeling tools: Roboflow, LabelImg, CVAT.

Run:
    python train_model.py
"""

from ultralytics import YOLO

BASE_MODEL = "yolov8n.pt"
DATA_YAML = "dataset/data.yaml"
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
PROJECT_NAME = "runs/detect"
RUN_NAME = "custom_motion_model"


def train():
    model = YOLO(BASE_MODEL)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=RUN_NAME,
    )
    print("\nTraining complete!")
    print(f"Best weights: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print("Ise motion_detector/config.py me MODEL_NAME = '<path>' set karke use karo.")


def validate(weights_path):
    model = YOLO(weights_path)
    print(model.val())


if __name__ == "__main__":
    train()
    # validate(f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
