import argparse
import csv
import os
import pathlib

import cv2
import ffmpeg
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

tqdm_bar_format = (
    "{desc}: {percentage:3.0f}% |{bar:30}| [{elapsed}<{remaining}, {rate_fmt}]"
)


def str2tuple(s):
    return tuple(map(int, s.split("x")))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MOT Challenge format with bounding boxes and trajectories."
    )
    parser.add_argument(
        "--mot-ch-file",
        "-m",
        type=str,
        required=True,
        help="Path to MOT Challenge format file.",
    )
    parser.add_argument(
        "--out-file",
        "-o",
        type=str,
        required=True,
        help="Path to output file without extension.",
    )
    parser.add_argument(
        "--image-dir",
        "-i",
        type=str,
        required=True,
        help="Path to directory containing images correspond to the MOT Challenge format file.",
    )
    parser.add_argument(
        "--resize",
        type=str2tuple,
        default=None,
        help="Resize images to the specified size. Example: 1920x1080",
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help="If true, saves all frames as an MP4 video; if false, saves only the final frame as a PNG image.",
    )
    parser.add_argument(
        "--show-bbox",
        action="store_true",
        help="If true, shows bounding boxes in the visualization.",
    )
    args = parser.parse_args()

    ch_file = args.mot_ch_file
    image_dir = args.image_dir
    out_file = args.out_file
    size = args.resize
    mp4 = args.mp4
    show_bbox = args.show_bbox

    sub = {}
    with open(ch_file, "r") as f:
        sub_csv = csv.reader(f)
        for line in sub_csv:
            frame_id = int(float(line[0]))
            if frame_id not in sub:
                sub[frame_id] = []
            sub[frame_id].append(
                {"track_id": int(float(line[1])), "bbox": list(map(float, line[2:6]))}
            )

    visualize(sub, image_dir, out_file, size, mp4, show_bbox)


def visualize(sub, image_dir, out_file, size=None, mp4=False, show_bbox=False):
    image_paths = sorted([str(x) for x in pathlib.Path(image_dir).rglob("*.jpg")])
    # target colors for different track ids
    custom_colors = [
        "#BB9727",  # 金黄色
        "#C76DA2",   # 粉紫色
        "#2455A4", # 蓝色
        "#FFE588",  # 深蓝色
        "#32B897",  # 青绿色
        "#FF9500",  # 浅黄色
        "#F6CAE5",  # 浅粉色
        "#009DFF",  # 天蓝色
        "#B288FF",  # 紫色
        "#22FF00",  # 绿色
        "#F27970",  # 珊瑚红
    ]
    colors = [hex_to_rgb(hex_color) for hex_color in custom_colors]
    bar = tqdm(total=len(image_paths), desc="Process Images", bar_format=tqdm_bar_format)
    drew_images = []
    trajectories = {}  # track_id -> [(frame_id, x, y), ...]
    trajectory_window = 10
    
    for i, image_path in enumerate(image_paths, 1):
        read_image = Image.open(image_path)
        draw = ImageDraw.Draw(read_image)
        annotations = sub.get(i, [])

        for annotation in annotations:
            track_id = annotation["track_id"]
            bbox = annotation["bbox"]

            color = colors[(track_id - 1) % len(colors)]

            if track_id not in trajectories:
                trajectories[track_id] = []
            # record the frame information along with tractory
            center = get_box_center(bbox)
            trajectories[track_id].append((i, center[0], center[1]))

            if show_bbox:
                draw.rectangle(
                    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    outline=color,
                    width=5,
                )

        # using recent $trajectory_window frames to draw trajectory
        for _track_id, trajectory in trajectories.items():
            if len(trajectory) > 1:
                # filter the trajectory to only include points within the window
                recent_trajectory = [
                    (x, y) for frame_id, x, y in trajectory 
                    if i - trajectory_window <= frame_id <= i
                ]
                
                if len(recent_trajectory) > 1:
                    draw.line(
                        recent_trajectory, 
                        fill=colors[(_track_id - 1) % len(colors)], 
                        width=5
                    )

        if size is not None:
            read_image = read_image.resize(size)

        drew_images.append(read_image)

        bar.update(1)

    bar.close()

    save(drew_images, out_file, mp4)


def mcolor2tuple(mcolor):
    return tuple(int(mcolors.hex2color(mcolor)[i] * 255) for i in range(3))


def get_box_center(bbox):
    return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)


def save(images, out_file, mp4):
    if mp4:
        file_with_ext = f"{out_file}.mp4"
        write_mp4(images, file_with_ext)
    else:
        file_with_ext = f"{out_file}.png"
        images[-1].save(file_with_ext)
    print(f"Saved to {file_with_ext}.")


def write_mp4(pil_images, file_name):
    bar = tqdm(
        total=len(pil_images) + 2, desc="Write to MP4  ", bar_format=tqdm_bar_format
    )
    tmp_file_name = f"{file_name}.tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(tmp_file_name, fourcc, 10, pil_images[0].size)
    for pil_image in pil_images:
        video.write(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
        bar.update(1)
    video.release()
    bar.update(1)
    ffmpeg.input(tmp_file_name).output(file_name, vcodec="libx264").run(
        overwrite_output=True, quiet=True
    )
    bar.update(1)
    os.remove(tmp_file_name)
    bar.close()


def hex_to_rgb(hex_color):
    """transform hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


if __name__ == "__main__":
    main()
