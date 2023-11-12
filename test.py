from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path
from glob import glob
import json
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import shutil
from PIL import Image

sample_list = ['000000', '000022', '000052', '000063', '000501']

valid_dir= "./valid/"

model = YOLO("./runs/detect/train5/weights/best.pt")
preds = model.predict(valid_dir, save=True)

for sample_name in sample_list:
    img_path = "./runs/detect/predict/" + sample_name + '.png'
    image = cv2.imread(img_path)
    cv2.imshow('Result: ', image)
    cv2.imwrite(valid_dir + sample_name + '.png', image)
    print("\n")
    cv2.waitKey(0)

