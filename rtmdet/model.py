# Inherit Base Configs
_base_ = [ './rtmdet-ins_tiny_8xb32-300e_coco.py' ]

# Define Integration with Tensorboard
# Reference:
# (https://blog.roboflow.com/how-to-train-rtmdet-on-a-custom-dataset/)
_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
    ]

# Model Fine-Tuning
# Reference:
# https://mmdetection.readthedocs.io/en/latest/user_guides/finetune.html
model = dict(bbox_head=dict(num_classes=2))

# Modify Dataset
dataset_type = 'CocoDataset'
classes = ('person', 'background')
data_root = '/../data/sard_yolo/'

# Declare Albumentations
transforms = [
    dict(type='HueSaturationValue'),
    dict(
        type='HueSaturationValue',
        sat_shift_limit=0,
        val_shift_limit=0
        ),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=0,
        val_shift_limit=0
        ),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=0,
        sat_shift_limit=0
        ),
    dict(
        type='Sharpen',
        p=0.8
        ),
    dict(type='Solarize'),
    dict(type='GaussianBlur')
    ]

# Set Up Train Pipeline
# Reference:
# https://github.com/open-mmlab/mmdetection/issues/11273
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args={{_base_.backend_args}}
        ),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=True
        ),
    dict(
        type='Albu',
        transforms=transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels'],
            min_visibility=0.0,
            filter_lost_elements=True
            ),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
            },
        skip_img_without_anno=True
        )
    ]

# Modify Dataloaders
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='ann_files/polygons/_train_annotations.coco.json',
        data_prefix=dict(img='images/train/')
        )
    )

train_dataloader['dataset']['pipeline'] = train_pipeline

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='ann_files/_valid_annotations.coco.json',
        data_prefix=dict(img='images/valid/')
        )
    )

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='ann_files/_test_annotations.coco.json',
        data_prefix=dict(img='images/test/')
        )
    )

# Modify Evaluators
test_evaluator = dict(
    ann_file=f'{data_root}ann_files/_test_annotations.coco.json',
    outfile_prefix='./work_dirs/rtmdet/test'
    )

val_evaluator = dict(ann_file=f'{data_root}ann_files/_valid_annotations.coco.json')

# Modify Training Schedule
optim_wrapper = dict(optimizer=dict(lr=0.001))

param_scheduler = [
    dict(
        eta_min=0.001 * 0.05,
        begin=40 // 2,
        end=40,
        T_max=40 // 2
        )
]

train_cfg = dict(
    max_epochs=40,
    val_interval=1
    )

default_hooks = dict(
    logger=dict(interval=1),
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=40
        )
    )

# Use Pre-Trained Model
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth' # noqa
