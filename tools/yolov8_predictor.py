import mmcv
import numpy as np
import torch
import os.path as osp
from loguru import logger
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict


class YOLOV8Predictor:
    def __init__(self, checkpoint_file, config=None, config_file=None, device='cuda:0', fp16=False, tta=False):
        loaded_config = None
        if config is not None:
            if not isinstance(config, Config):
                raise TypeError(f"Expected 'config' to be of type mmengine.config.Config, but got {type(config)}")
            loaded_config = config
            logger.info("Using provided Config object.")
        elif config_file is not None:
            logger.info(f"Loading config from file: {config_file}")
            loaded_config = Config.fromfile(config_file)
        else:
            raise ValueError("Either 'config' (mmengine.config.Config object) or 'config_file' (path string) must be provided to YOLOV8Predictor.")

        if tta:
            if loaded_config is None:
                 raise ValueError("Config must be loaded before applying TTA settings.")
            loaded_config.model = ConfigDict(**loaded_config.tta_model, module=loaded_config.model)
            test_data_cfg = loaded_config.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            if 'batch_shapes_cfg' in test_data_cfg:
                test_data_cfg.batch_shapes_cfg = None
            test_data_cfg.pipeline = loaded_config.tta_pipeline
        self.tta = tta
        
        if loaded_config is None:
            raise ValueError("Config must be loaded before initializing the model.")
            
        self.model = init_detector(loaded_config, checkpoint_file, device=device, cfg_options={})
        if fp16:
            self.model = self.model.half()
        self.model.eval()

    def inference(self, img, timer):
        """
        Args:
            img: image path or numpy array
            timer: Timer object for inference time measurement
        Returns:
            outputs: List[numpy.ndarray], each element contains [x1,y1,x2,y2,score,class_id]
            img_info: dict, including: 'id', 'file_name', 'raw_img'
        """
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img_info["raw_img"] = mmcv.imread(img)
        else:
            img_info["file_name"] = None
            img_info["raw_img"] = img

        timer.tic()
        result = inference_detector(self.model, img_info["raw_img"])

        # output format ajustment
        pred_instances = result.pred_instances
        if len(pred_instances.bboxes) == 0:
            outputs = [None]
            logger.info("No detections found")
        else:
            outputs = [np.concatenate([
                pred_instances.bboxes.cpu().numpy(),
                pred_instances.scores.cpu().numpy()[:, None],
            ], axis=1)]

        return outputs, img_info
