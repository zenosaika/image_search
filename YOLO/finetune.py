import os
from ultralytics import YOLO


WEIGHT = 'best.pt'
DATASET = 'augmented_dataset'
EPOCH = 1


# Load weight
if os.path.exists(WEIGHT):
    model = YOLO(WEIGHT)
    print('[WEIGHT] using best.pt')
else:
    model = YOLO('YOLOv8x-cls.pt')
    print('[WEIGHT] using YOLOv8x-cls.pt')

# Start training
model.train(data=DATASET, epochs=EPOCH, project='runs')
