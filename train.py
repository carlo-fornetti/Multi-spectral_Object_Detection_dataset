from ultralytics import YOLO
import numpy as np
from pathlib import Path
from glob import glob
import json
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import shutil
from PIL import Image

base_dir= Path("./dataset/ir_det_dataset")
img_train_path = base_dir /'Images'/'fir'
label_path = base_dir /'labels'/'fir'

ims = sorted(list(img_train_path.glob('*')))
labels = sorted(list(label_path.glob('*')))
pairs = list(zip(ims,labels))

train, test = train_test_split(pairs, test_size=0.1,shuffle=True)

train_path = Path('train').resolve()
train_path.mkdir(exist_ok=True)
valid_path = Path('valid').resolve()
valid_path.mkdir(exist_ok=True)

for t_img, t_lb in tqdm(train):
    im_path = train_path / t_img.name
    lb_path = train_path / t_lb.name
    shutil.copy(t_img,im_path)
    shutil.copy(t_lb,lb_path)

for t_img, t_lb in tqdm(test):
    im_path = valid_path / t_img.name
    lb_path = valid_path / t_lb.name
    shutil.copy(t_img,im_path)
    shutil.copy(t_lb,lb_path)

model = YOLO("yolov8n.pt")
train = model.train(data="data.yml", epochs=10)

valid = model.val()