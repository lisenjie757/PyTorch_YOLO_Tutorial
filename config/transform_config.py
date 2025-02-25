# transform config

# ----------------------- YOLOv5-Style -----------------------
yolov5_huge_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 0.0,
    'translate': 0.2,
    'scale': 0.9,
    'shear': 0.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 0.2,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

yolov5_large_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 0.0,
    'translate': 0.2,
    'scale': 0.9,
    'shear': 0.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 0.15,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

yolov5_medium_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 0.0,
    'translate': 0.2,
    'scale': 0.9,
    'shear': 0.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 0.10,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

yolov5_small_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 0.0,
    'translate': 0.2,
    'scale': 0.9,
    'shear': 0.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 0.05,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

yolov5_tiny_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 1.0,
    'mixup_prob': 0.05,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

yolov5_nano_trans_config = {
    'aug_type': 'yolov5',
    # Basic Augment
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Mosaic & Mixup
    'mosaic_prob': 0.5,
    'mixup_prob': 0.0,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp
}

# ----------------------- SSD-Style -----------------------
ssd_trans_config = {
    'aug_type': 'ssd',
    # Mosaic & Mixup are not used for SSD-style augmentation
    'mosaic_prob': 0.,
    'mixup_prob': 0.,
    'mosaic_type': 'yolov5_mosaic',
    'mixup_type': 'yolov5_mixup',
    'mixup_scale': [0.5, 1.5]
}
