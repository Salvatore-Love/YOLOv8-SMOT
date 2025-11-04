import os
import os.path as osp
import sys
# Add the project root to sys.path to allow importing 'trackers'
_CURRENT_FILE_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.dirname(_CURRENT_FILE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from pathlib import Path
import time
import numpy as np

import torch
from loguru import logger

from yolov8_predictor import YOLOV8Predictor
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.tracking_utils.timer import Timer

import cv2
from mmengine.config import Config
from tools.frame_utils import adjust_frame_dimensions

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

def is_image(path):
    return isinstance(path, str) and path.lower().endswith(IMG_EXTENSIONS)

def is_video(path):
    return isinstance(path, str) and path.lower().endswith(VIDEO_EXTENSIONS)

from utils.args import make_parser


def get_video_image_dict(root_path):
    video_image_dict = {}
    
    for video_name in os.listdir(root_path):
        video_path = osp.join(root_path, video_name)
        
        if not osp.isdir(video_path):
            continue

        image_paths = []
        for maindir, _, file_name_list in os.walk(video_path):
            for filename in file_name_list:
                ext = osp.splitext(filename)[1].lower()
                if ext in IMG_EXTENSIONS:
                    image_paths.append(osp.join(maindir, filename))

        video_image_dict[video_name] = sorted(image_paths)

    return video_image_dict

def frame_iterator(source):
    if isinstance(source, list): # Image sequence
        for frame_id, img_path in enumerate(source, 1):
            yield frame_id, img_path
    elif is_video(source): # Video file
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {source}")
            return
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            yield frame_id, frame
        cap.release()

def predict_videos(base_config_file, base_checkpoint_file, res_folder, args):
    # Setup result folders
    if hasattr(args, 'save_path') and args.save_path:
        current_res_folder = osp.join(res_folder, args.save_path)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        current_res_folder = osp.join(res_folder, f"{timestamp}")
    
    tracks_folder = osp.join(current_res_folder, "tracks")
    dets_folder = osp.join(current_res_folder, "dets")
    os.makedirs(tracks_folder, exist_ok=True)
    os.makedirs(dets_folder, exist_ok=True)

    # Determine if the path is a single video file or a directory of sequences
    video_sources = {}
    if is_video(args.path):
        video_name = Path(args.path).stem
        video_sources = {video_name: args.path}
    elif osp.isdir(args.path):
        # Check if the directory itself contains images
        image_files = sorted([f for f in os.listdir(args.path) if is_image(osp.join(args.path, f))])
        if image_files:
            video_name = Path(args.path).name
            video_sources = {video_name: [osp.join(args.path, f) for f in image_files]}
        else:
            # Otherwise, assume it's a directory of video subdirectories
            video_sources = get_video_image_dict(args.path)
    else:
        raise ValueError(f"Input path {args.path} is not a valid video file or directory.")

    if not video_sources:
        logger.error(f"No valid video sources found in {args.path}. Please check the path and directory structure.")
        return

    for video_name, source in video_sources.items():
        logger.info(f"Processing: {video_name}")

        tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte, delta_t=args.deltat, min_hits=args.min_hits, asso_func=args.asso, iou_thresh_decrease=args.iou_thresh_decrease, track_thresh_low=args.track_thresh_low)
        timer = Timer()
        track_results = []
        det_results = []

        if hasattr(args, 'use_saved_dets') and args.use_saved_dets:
            saved_det_file = osp.join(args.saved_dets_path, f"{video_name}.txt")
            if not osp.exists(saved_det_file):
                logger.error(f"Detection file not found: {saved_det_file}")
                continue
            
            with open(saved_det_file, 'r') as f:
                lines = f.readlines()
                frame_dets = {}
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) < 6: # Min: frame_id,x1,y1,x2,y2,conf
                        logger.warning(f"Skipping malformed line in {saved_det_file}: {line.strip()}")
                        continue
                    frame_id, x1, y1, x2, y2, conf = parts[:6]
                    frame_id = int(frame_id)
                    if frame_id not in frame_dets:
                        frame_dets[frame_id] = []
                    det = np.array([float(x1), float(y1), float(x2), float(y2), float(conf)])
                    frame_dets[frame_id].append(det)

            for frame_id in sorted(frame_dets.keys()):
                timer.tic()
                output_results = np.array(frame_dets[frame_id])
                online_targets = tracker.update(output_results)
                if online_targets is not None:
                    for t in online_targets:
                        tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                        tid = t[4]
                        if tlwh[2] * tlwh[3] > args.min_box_area:
                            track_results.append(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1,1,1\n")
                timer.toc()
                if frame_id % 100 == 0:
                    logger.info(f'Processing frame {frame_id} ({1. / max(1e-5, timer.average_time):.2f} fps)')
        else:
            # This block runs when not using saved detections
            loaded_cfg_for_video = None
            frame_width, frame_height = None, None

            if isinstance(source, list) and source: # Image sequence
                img = cv2.imread(source[0])
                if img is not None: frame_height, frame_width = img.shape[:2]
            elif is_video(source): # Video file
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

            if frame_width and frame_height:
                # Adjust config of detector based on frame size
                try:
                    target_width, target_height = adjust_frame_dimensions(frame_width, frame_height)
                    img_scale = (target_width, target_height)
                    cfg = Config.fromfile(base_config_file)
                    if hasattr(cfg, 'test_dataloader') and 'pipeline' in cfg.test_dataloader.dataset:
                        for transform in cfg.test_dataloader.dataset.pipeline:
                            if transform.get('type') in ['YOLOv5KeepRatioResize', 'LetterResize']:
                                transform['scale'] = img_scale
                    if hasattr(cfg, 'model') and 'data_preprocessor' in cfg.model and cfg.model.data_preprocessor and 'batch_augments' in cfg.model.data_preprocessor:
                        for aug in cfg.model.data_preprocessor.batch_augments:
                            if aug.get('type') == 'BatchFixedSizePad': aug['size'] = img_scale
                    loaded_cfg_for_video = cfg
                    logger.info(f"Adjusted config for {video_name} to target size: {img_scale}")
                except Exception as e:
                    logger.error(f"Error adjusting config for {video_name}: {e}. Using base config.")
            
            current_predictor = YOLOV8Predictor(
                config=loaded_cfg_for_video or base_config_file,
                checkpoint_file=base_checkpoint_file,
                device=args.device,
                fp16=args.fp16,
                tta=getattr(args, 'tta', False)
            )

            for frame_id, frame_data in frame_iterator(source):
                timer.tic()
                outputs, _ = current_predictor.inference(frame_data, timer)
                
                online_targets = None
                if outputs[0] is not None:
                    output_results = outputs[0]
                    for det_item in output_results:
                        x1, y1, x2, y2, conf = det_item[:5]
                        det_results.append(f"{frame_id},{x1},{y1},{x2},{y2},{conf}\n")
                    online_targets = tracker.update(output_results)

                if online_targets is not None:
                    for t in online_targets:
                        tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                        tid = t[4]
                        if tlwh[2] * tlwh[3] > getattr(args, 'min_box_area', 10):
                            track_results.append(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1,1,1\n")
                timer.toc()
                if frame_id % 20 == 0:
                    logger.info(f'Processing frame {frame_id} ({1. / max(1e-5, timer.average_time):.2f} fps)')

        # Save results
        if getattr(args, 'save_result', False):
            track_file = osp.join(tracks_folder, f"{video_name}.txt")
            with open(track_file, 'w') as f:
                f.writelines(track_results)
            logger.info(f"Saved tracking results to {track_file}")
            
            if not getattr(args, 'use_saved_dets', False):
                det_file = osp.join(dets_folder, f"{video_name}.txt")
                with open(det_file, 'w') as f:
                    f.writelines(det_results)
                logger.info(f"Saved detection results to {det_file}")

def main(args):
    if not args.path:
        raise ValueError("Please specify the path to the video file or image sequence directory")

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        res_folder = osp.join(output_dir, "predictions", Path(args.path).stem)
        os.makedirs(res_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    config_file = "mmyolo/configs/yolov8/yolov8_l.py"
    checkpoint_file = 'checkpoint/yolov8_l.pth' 
    
    # Allow overriding checkpoint via args for the base checkpoint_file
    base_checkpoint_file = getattr(args, 'checkpoint', checkpoint_file) # Use the one defined earlier or from args
    if hasattr(args, 'checkpoint_file') and args.checkpoint_file:
        base_checkpoint_file = args.checkpoint_file
    
    # Ensure res_folder is defined if not args.save_result
    if not hasattr(args, 'save_result') or not args.save_result:
        if 'res_folder' not in locals():
            logger.warning("res_folder not explicitly set in main when save_result is False. Defaulting to output_dir for predict_videos.")
            res_folder = output_dir

    predict_videos(config_file, base_checkpoint_file, res_folder, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
