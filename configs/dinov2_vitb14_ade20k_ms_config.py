dataset_type = 'ADE20KDataset'
data_root = '/home/ADEChallengeData2016/ADEChallengeData2016'
crop_size = (518, 518)
stride = (crop_size[0] // 2, crop_size[1] // 2)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53,],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    std=[58.395, 57.12, 57.375,],
    type='SegDataPreProcessor')
default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=20, type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=True, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=100))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios=[1.0, 1.32, 1.73],   #, 2.28, 3.],
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True)
norm_cfg = dict(requires_grad=True, type='SyncBN')
train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=20)
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999))
optim_wrapper = dict(
    clip_grad=None,
    optimizer=optimizer,
    type='OptimWrapper')
param_scheduler = [                 #NOTE: this definition is slightly different from the original config
    dict(
        type='PolyLR',
        power=1.0,
        begin=0,
        end=train_cfg['max_epochs'],
        eta_min=0.0,
        by_epoch=True,
    )
]
resume = True
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RandomResize', scale=(99999999, 518), ratio_range=(1.0, 3.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        data_root=data_root,
        pipeline=train_pipeline,
        type=dataset_type),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
test_cfg = dict(type='TestLoop')
test_evaluator = dict(iou_metrics=['mIoU',], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(99999999, 518), keep_ratio=True),           
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    # dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.25, 1.0), keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.),
    dict(type='PackSegInputs'),
]
test_dataloader = dict(
    batch_size=1, # //4 if tta
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(
            img_path='images/validation', seg_map_path='annotations/validation'),
        data_root=data_root,
        pipeline=test_pipeline),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_cfg = dict(type='ValLoop')
val_evaluator = dict(iou_metrics=['mIoU',], type='IoUMetric')
val_dataloader = test_dataloader
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=img_ratios[0][0], keep_ratio=True),
                dict(type='Resize', scale_factor=img_ratios[0][1], keep_ratio=True),
                dict(type='Resize', scale_factor=img_ratios[0][2], keep_ratio=True),
                # dict(type='Resize', scale_factor=img_ratios[0][3], keep_ratio=True),
                # dict(type='Resize', scale_factor=img_ratios[0][4], keep_ratio=True),
            ],
            [
                dict(type='Pad', size=crop_size),
            ],
            [
                dict(type='RandomFlip', direction='horizontal', prob=0.0),
                dict(type='RandomFlip', direction='horizontal', prob=1.0),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ])
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='DinoVisionBackbone',
        size='base',
        img_size=crop_size,
        patch_size=14,
        freeze_vit=True,
        init_cfg=dict(type='Pretrained', checkpoint="../dinov2/pretrained_models/dinov2_vitb14_pretrain.pth"),
        # norm_cfg=norm_cfg,
        # out_indices=[8, 9, 10, 11]
        ),
    decode_head=dict(
        type='BNHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        input_transform='resize_concat',
        channels=3072,
        dropout_ratio=0,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=stride),
    data_preprocessor=data_preprocessor)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=vis_backends)