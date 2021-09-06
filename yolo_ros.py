#!/usr/bin/env python3

import torch
from models.experimental import attempt_load
from utils.torch_utils import select_device

weights = 'best.pt'
# Model
device = select_device("")
model = attempt_load(weights, map_location=device)
# torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = '/home/lukez/COHIRNT/data/san_diego_testing/data_collection/july_16_images/test_1/frame0001.png'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

