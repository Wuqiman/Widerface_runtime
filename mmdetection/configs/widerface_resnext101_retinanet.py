# model settings
#fp16 = dict(loss_scale=512.)
model = dict(
    type='RetinaNet',
    pretrained=None,
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        dcn=dict(
            modulated=True,
            groups=64,
            deformable_groups=1,
            fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    # backbone=dict(
    #     type='ResNext',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=False),
    #     norm_eval=True,
    #     style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='GARetinaHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        octave_ratios=[0.5, 1.0, 2.0],
        #anchor_strides=[8, 16, 32, 64, 128],
        anchor_strides=[4, 8, 16, 32, 64],
        anchor_base_sizes=None,
        anchoring_means=[.0, .0, .0, .0],
        anchoring_stds=[1.0, 1.0, 1.0, 1.0],
        target_means=(.0, .0, .0, .0),
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),

        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        #loss_shape=dict(
        #    type='IoULoss',
        #    #use_bounded=True,
        #    #beta=0.2,
        #    loss_weight=1.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.04, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    single_stage=dict(
        ga_assigner=dict(
            type='ApproxMaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.4,
            ignore_iof_thr=-1),
        ga_sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        center_ratio=0.2,
        ignore_ratio=0.5,
        debug=False))
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    #max_per_img=750)
    max_per_img=200)
#test_cfg = dict(
#    single_stage=dict(
#        nms_pre=1000,
#        min_bbox_size=0,
#        score_thr=0.05,
#        nms=dict(type='nms', iou_thr=0.5),
#        max_per_img=750))
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/WIDER_FACE2019/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
data = dict(
imgs_per_gpu=4,
workers_per_gpu=4,
train=dict(
    type=dataset_type,
    ann_file=data_root + 'WIDER_train/instances_train2019.json',
    img_prefix=data_root + 'WIDER_train/images/',
    img_scale=(640, 640),
    img_norm_cfg=img_norm_cfg,
    size_divisor=32,
    flip_ratio=0.5,
    with_mask=False,
    with_crowd=True,
    with_label=True,
    extra_aug=dict(
        photo_metric_distortion=dict(
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18),
        random_crop_face=dict(min_crop_size=0.3)),
    resize_keep_ratio=True),
val=dict(
    type=dataset_type,
    ann_file=data_root + 'WIDER_val/instances_val2019.json',
    img_prefix=data_root + 'WIDER_val/images/',
    img_scale=(2100, 800),
    img_norm_cfg=img_norm_cfg,
    size_divisor=32,
    flip_ratio=0,
    with_mask=False,
    with_crowd=True,
    with_label=True),

test=dict(
    type=dataset_type,
    ann_file=data_root + 'WIDER_val/instances_val2019.json',
    img_prefix=data_root + 'WIDER_val/images/',
    img_scale=(2100, 800),
    #(500, 800, 1100, 1400, 1700
    #img_scale=[(1000,500),(1200, 800),(1500,1100),(1700,1400),(2100,1700)],
    img_norm_cfg=img_norm_cfg,
    size_divisor=32,
    flip_ratio=0,
    with_mask=False,
    with_label=False,
    test_mode=True))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001,
                 auto_adjust=True, base_batch_size=16)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[11, 17])
 
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:disable
total_epochs = 21
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/widerface_ga_retinanet_x101_64x4d_21e'

#load_from = dict(filename='./work_dirs/widerface_ga_retinanet_x101_64x4d_21e/cascade_rcnn_epoch_21.pth',exclude_fields=['neck','head'])
#load_from = dict(filename='./work_dirs/widerface_ga_retinanet_x101_64x4d_21e/cascade_rcnn_epoch_21.pth',exclude_fields=['neck','head'])
load_from=None
resume_from = None
workflow = [('train', 1)]




