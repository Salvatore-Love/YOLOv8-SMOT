import os
os.getcwd()

# from sahi.utils.yolov8 import (
#     download_yolov8s_model,
# )

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
# from IPython.display import Image

model_type = "yolov5"
model_path = "/media/2020/AIBSc002_gliang/LG/bird/mmyolo/work_dirs/yolov5_s-v61_syncbn_8xb16-300e_bird/best_coco/bbox_mAP_epoch_200.pth "
model_device = "cuda:0,1" # or 'cuda:0'
model_confidence_threshold = 0.2

slice_height = 1280
slice_width = 1280
overlap_height_ratio = 0.3
overlap_width_ratio = 0.3

source_image_dir = "/media/2020/AIBSc002_gliang/LG/bird/data/MVA2023_Challenge/data/mva2023_sod4bird_pub_test/images/"
dataset_json_path = "/media/2020/AIBSc002_gliang/LG/bird/data/MVA2023_Challenge/data/mva2023_sod4bird_pub_test/annotations/public_test_coco_empty_ann.json"

#%%

predict(
    model_type=model_type,
    model_path=model_path,
    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source=source_image_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
    dataset_json_path = dataset_json_path,
    novisual = True,

)