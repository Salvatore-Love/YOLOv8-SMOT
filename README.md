# Project Name - Multi-Object Tracking Inference

This repository provides the official Pytorch implementation of [YOLOv8-SMOT: An Efficient and Robust Framework for Real-Time Small Object Tracking via Slice-Assisted Training and Adaptive Association](https://arxiv.org/abs/2507.12087) for performing multi-object tracking inference, which won the **Best Solution Award** of [SMOT4SB Challenge 2025](https://www.mva-org.jp/mva2025/challenge). 

[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2507.12087-B31B1B.svg)](https://arxiv.org/abs/2507.12087)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/abandoned/YOLOv8-SMOT)

It is focused exclusively on the inference process, utilizing pre-trained models to track objects in videos or image sequences. The model is based on the repositories of [mmyolo](https://github.com/open-mmlab/mmyolo) and [OCSORT](https://github.com/noahcao/OC_SORT).

## Performance
SMOT4SB
| Model | SO-HOTA | SO-DetA | SO-AssA | Param.(M) |
| :-- | :-: | :-: | :-: | :-: |
| [**YOLOv8-SMOT-L**](https://huggingface.co/abandoned/YOLOv8-SMOT/resolve/main/yolov8_l.pth?download=true) | **55.205** | **51.716** | **59.082**   | **43.7** |
| [**YOLOv8-SMOT-M**](https://huggingface.co/abandoned/YOLOv8-SMOT/resolve/main/yolov8_m.pth?download=true) | **54.426** | **49.529** | **59.962**  | **25.9** |
| [**YOLOv9-SMOT-S**](https://huggingface.co/abandoned/YOLOv8-SMOT/resolve/main/yolov8_s.pth?download=true) | **53.808** | **48.388** | **59.979**   | **11.2** |

## Quick Start

This section provides instructions on how to run inference using the [`tools/predict.py`](tools/predict.py:0) script. This repository is focused exclusively on inference.

### 1. Environment Setup

This section guides you through setting up the necessary environment to run the multi-object tracking inference scripts. It is crucial to follow these steps precisely to ensure compatibility and reproducibility.

**Step 1: Create and Activate a Conda Environment**

We strongly recommend creating a dedicated Conda environment. This isolates the project's dependencies and avoids conflicts with other Python projects.

```bash
# You can replace 'mmyolo_env' with your preferred environment name.
# The original environment for this project used Python 3.8.12.
conda create -n mmyolo_env python=3.8.12 -y
conda activate mmyolo_env
```
**Step 2: Install PyTorch**

PyTorch is a core dependency. The specific version might depend on your hardware (especially CUDA version). This project was tested with PyTorch 1.12.1 and CUDA 11.3.

```bash
# Command for installing PyTorch 1.12.1 with CUDA 11.3:
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
<!-- *   If you have a different CUDA version, or if you need a CPU-only version, please visit the [official PyTorch website (previous versions)](https://pytorch.org/get-started/previous-versions/) to find the appropriate installation command for your system. -->


**Step 3: Install MMYOLO and its Dependencies (Following `mmyolo/README_Ours.md`)**

To ensure MMYOLO is installed correctly for this project, we will follow the installation steps similar to those outlined in [`mmyolo/README_Ours.md`](mmyolo/README_Ours.md:0). This involves installing `openmim` and then using it to install MMYOLO and its specific dependencies.

1.  **Install `openmim`:**
    `openmim` is the official package manager for OpenMMLab projects.
    ```bash
    pip install -U openmim
    ```

2.  **Install MMYOLO Core Dependencies using `mim`:**
    This step installs the necessary packages listed in [`mmyolo/requirements/mminstall.txt`](mmyolo/requirements/mminstall.txt), which provides a comprehensive set of dependencies for MMYOLO.
    ```bash
    # Ensure you are in the root directory of this repository.
    mim install -r mmyolo/requirements/mminstall.txt
    ```
    <!-- *Note: The file [`mmyolo/requirements/albu.txt`](mmyolo/requirements/albu.txt) is typically for training and might not be strictly needed for pure inference. If you encounter issues or plan to extend to training, you might consider installing it: `mim install -r mmyolo/requirements/albu.txt`.* -->

3.  **Install the MMYOLO Package (Editable Mode):**
    This command installs the `mmyolo` package itself from the local `mmyolo` subdirectory. The `-e` flag installs it in "editable" mode, meaning changes in the `mmyolo` source directory will be reflected in the installed package.
    ```bash
    # Ensure you are in the root directory of this repository.
    mim install -v -e ./mmyolo
    ```
    

**Step 4: Install All Other Project Dependencies (including Trackers like OCSORT)**

After successfully installing PyTorch, MMYOLO, and its specific dependencies, the next crucial step is to install all other packages required for the complete functionality of this project. These are listed in the root [`requirements.txt`](requirements.txt) file. This file includes dependencies for various tracking algorithms (e.g., OCSORT, ByteTrack), data processing utilities, and any other libraries essential for running the inference scripts. And some dependencies of included in this file have been installed in the former steps.

```bash
# Ensure you are in the root directory of this cloned repository.
# This command installs all remaining dependencies specified in requirements.txt.
pip install -r requirements.txt
```

**Important Notes:**
*   The installation sequence (Step 1: Conda Env -> Step 2: PyTorch -> Step 3: MMYOLO via `mim` -> Step 4: root [`requirements.txt`](requirements.txt)) is designed to establish a stable and complete environment. Following this order helps minimize potential dependency conflicts.
*   The root [`requirements.txt`](requirements.txt) file is critical for replicating the full project environment. It complements the MMYOLO-specific dependencies by providing all other necessary packages, ensuring that components like OCSORT and other utilities are correctly installed with their required versions.
*   The setup steps can create an similar environment with our develop and test one's, but it still leads to some subtle differences of detection results and thus may diff from the performance we submitted during pub-test phase. Since the limitation of evaluation on test data, the effect of these changes of detection results is unknown.

With these steps completed, your environment should be ready for configuring the model and running inference as described in the subsequent sections.

### 2. Configure Model and Checkpoint in `tools/predict.py`

The [`tools/predict.py`](tools/predict.py) script requires manual configuration for the model configuration file and the checkpoint file.

- **Model Configuration File (`config_file`):**
  Open [`tools/predict.py`](tools/predict.py) and locate the `config_file` variable within the `main()` function. You need to set this path to your desired model configuration. The script currently defaults to:
  `"mmyolo/configs/yolov8/yolov8_l.py"`
  Ensure this path points to the correct `.py` configuration file for your YOLOv8 Large model.

- **Model Checkpoint File (`checkpoint_file`):**
  Similarly, locate the `checkpoint_file` variable in [`tools/predict.py`](tools/predict.py) (within `main()`). You **must** change this path to:
  `"path_to_checkpoint/yolov8_l.pth"`
  Ensure the `path_to_checkpoint/yolov8_l.pth` file is present in your workspace.

<!-- - **Tracker Image Size (`args.img_size`):**
  Note that in [`tools/predict.py`](tools/predict.py), `args.img_size` is hardcoded to `(3840, 2176)`. This value is used by the OC-SORT tracker during the `tracker.update()` call, likely as a reference for the original video/image dimensions. The actual inference image size for the model is determined by the model's configuration file. -->

### 3. Running Inference

Once the environment is set up and [`tools/predict.py`](tools/predict.py) has been configured with the correct paths, you can run inference. The script processes video frames or image sequences from a specified input directory.

**General Command Structure:**

```bash
python tools/predict.py --path <path_to_input_directory_or_video> [OPTIONS]
```

**Key Arguments (from `utils/args.py`):**

*   `--path <string>`: **(Required)** Path to the directory containing video frame sequences (e.g., one subdirectory per video, each containing sorted image frames) or a directory containing ordered images of one video frames or a direct path to a video file if the script is adapted for it. The current `predict.py` expects a directory of image sequences.
*   `--device <string>`: Device to run the model on. Can be "gpu" or "cpu". (Default: "gpu")
*   `--save_path <string>`: Custom name for the subfolder where results will be saved. If not provided, a timestamp-based name is used. (Default: None)
*   `--fp16`: Use mixed-precision (FP16) for inference. (Default: False)
*   `--tta`: Use Test Time Augmentation. (Default: False)

**Tracking Parameters (OC-SORT):**

*   `--track_thresh <float>`: Detection confidence threshold for initializing a track. (Default: 0.25)
*   `--track_thresh_low <float>`: Lower detection confidence threshold for ByteTrack association. (Default: 0.1)
*   `--iou_thresh <float>`: IoU threshold for matching detections to existing tracks in SORT. (Default: 0.25)
*   `--iou_thresh_decrease <float>`: Factor to decrease IoU threshold in ByteTrack for low-score detections. (Default: 0.08)
*   `--min_hits <int>`: Minimum number of hits (frames) to confirm a track. (Default: 1)
*   `--deltat <int>`: Time step difference used for estimating direction/velocity. (Default: 1)
*   `--asso <string>`: Association/similarity function (e.g., "iou", "giou", "diou", "diou_expand"). (Default: "diou_expand")
*   `--use_byte`: Enable ByteTrack association mechanism. (Default: True, enabled if flag not explicitly set to false by modifying script)
*   `--min-box-area <float>`: Minimum bounding box area. Detections/tracks smaller than this will be filtered out. (Default: 10)

**Using Pre-computed Detections:**

*   `--use_saved_dets`: If specified, the script will use previously saved detection results instead of running the detector.
*   `--saved_dets_path <string>`: Path to the directory containing the saved detection files (e.g., `dets/`). (Default: "")

**Example Command:**

Assuming your input image sequences are in `/data/private_test/` (e.g., `/data/private_test/*/*.jpg` or `/data/private_test/*.jpg`), and you want to save results using default tracking parameters on the GPU:

```bash
python tools/predict.py \
    --path /data/private_test/ \
    --save_result \
    --device gpu
```

To use specific tracking thresholds and save to a custom-named output subfolder:

```bash
python tools/predict.py \
    --path /data/my_videos/ \
    --save_result \
    --save_path yolov8l_custom_run \
    --track_thresh 0.25 \
    --track_thresh_low 0.1 \
    --iou_thresh 0.25 \
    --iou_thresh 0.25 \
    --min-box-area 10 \
    --iou_thresh_decrease 0.08 \
    --use_byte \
    --asso diou_expand \
    --min_hits 1 \
    --deltat 1 \
    --min-box-area 10
```

You can use single video as input by replacing the value of `path` with `/data/video.mp4`.

### 4. Output Details

If `--save_result` is enabled:
*   Detection results (if not using `--use_saved_dets`) are saved in `results/predictions/<input_folder_name_or_save_path>/dets/<video_name>.txt`.
*   Tracking results are saved in `results/predictions/<input_folder_name_or_save_path>/tracks/<video_name>.txt`. **This is the track results saved in .txt file for evaluation.**

The format of these files is typically:
*   Detections: `frame_id, x1, y1, x2, y2, conf`
*   Tracks: `frame_id, track_id, bb_left, bb_top, bb_width, bb_height, 1, 1, 1` (MOT format)

Please check the [`tools/predict.py`](tools/predict.py) script for the exact output directory structure and file formats.

### 5. Visualization
We modified the code of [SMOT4SB Baseline](https://github.com/IIM-TTIJ/MVA2025-SMOT4SB) to visualize the tracking results: 

```bash
python3 tools/visualize_for_mot_ch.py -m track_result.txt -o visualized_video -i corresponding_video_folder --mp4 --show-bbox
```

## Citation
```
@ARTICLE{2025arXiv250712087Y,
  title = "{YOLOv8-SMOT: An Efficient and Robust Framework for Real-Time Small Object Tracking via Slice-Assisted Training and Adaptive Association}",
  author = {{Yu}, Xiang and {Liu}, Xinyao and {Liang}, Guang},
journal = {arXiv e-prints},
  year = 2025,
```

## Acknowledgement
<details><summary> <b>Expand</b> </summary>

* [https://www.mva-org.jp/mva2025/challenge](https://www.mva-org.jp/mva2025/challenge)
* [https://github.com/IIM-TTIJ/MVA2025-SMOT4SB](https://github.com/IIM-TTIJ/MVA2025-SMOT4SB)
* [https://github.com/open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo)
* [https://github.com/noahcao/OC_SORT](https://github.com/noahcao/OC_SORT)

</details>