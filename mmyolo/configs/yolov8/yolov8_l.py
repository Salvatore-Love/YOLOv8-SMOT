_base_ = './yolov8_m.py'

# ========================modified parameters======================
deepen_factor = 1.00
widen_factor = 1.00
last_stage_out_channels = 512

mixup_prob = 0.15

# =======================Unmodified in most cases==================
pre_transform = _base_.pre_transform
pre_transform = _base_.pre_transform
last_transform = _base_.last_transform

model = dict(
    backbone=dict(
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels]),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=[256, 512, last_stage_out_channels])))


