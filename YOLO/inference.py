import os
import pandas as pd
from ultralytics import YOLO


WEIGHT = 'runs/classify/train15/weights/best.pt'
CONFIDENCE = 0.55
OUTPUT = 'submission.csv'


# Load weight
if os.path.exists(WEIGHT):
    model = YOLO(WEIGHT)
    print('[WEIGHT] using best.pt')
else:
    model = YOLO('YOLOv8x-cls.pt')
    print('[WEIGHT] using YOLOv8x-cls.pt')


# load submission template (sample_submission.csv)
submission = pd.read_csv('sample_submission.csv')
submission['YOLO'] = 22 # set all row to 22 (class: others)


prefered_classes = [str(i) for i in range(22)] # 0 - 21


# loop through each line in sample_submission.csv
for idx, row in submission.iterrows():

    path_to_img = os.path.join('test/images', row['img_file'])
    result = model([path_to_img])[0]

    classnames = result.names
    probs = result.probs

    top5 = probs.top5
    top5conf = probs.top5conf

    predicted_class = classnames[top5[0]]
    confidence = top5conf[0]

    if confidence > CONFIDENCE and predicted_class in prefered_classes:
        submission.at[idx, 'YOLO'] = int(predicted_class)

# Save submission.csv
sub = submission[['img_file',]]
sub['class'] = submission['YOLO']
sub.to_csv(OUTPUT, index=False)
