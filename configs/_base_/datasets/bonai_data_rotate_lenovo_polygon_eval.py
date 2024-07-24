# encoding:utf-8
# 已经旋转过的数据
dataset_type = 'BONAI'
data_root = 'D:\\Documents\\Dataset\\building_footprint\\BONAI\\'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# dataset初始化阶段从json中读取图片文件信息和一些标注信息，在pipline中的中读取图片文件，在中的奖标注信息加入训练数据流，
# 然后进行一些增广和一致化操作，最后将数据流收集到一个字典中输出，字典的键即是collect中的几个
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', 
         with_bbox=True,# 可以collect：gt_bboxes,gt_labels
         with_mask=True,# 可以collect到：gt_masks
         with_offset=True),# 可以collect到：gt_offsets
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_offsets']),
    # 数据的输出与模型的forward_train输入参数一致。
    # 例如TwoStageDetector.forward_train(self,img,img_metas,gt_bboxes,gt_labels,gt_bboxes_ignore=None,gt_masks=None,proposals=None,**kwargs):
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),# 不会生效的，因为MultiScaleFlipAug中会在result中事先添加_results['flip'] = False
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_anno_pipeline = [
    dict(type='LoadAnnotations',
         with_bbox=True,# 可以collect：gt_bboxes,gt_labels
         with_mask=True,# 可以collect到：gt_masks
         with_offset=True,# 可以collect到：gt_offsets
         with_pitch_ratio=True),
    dict(type='Collect', keys=['gt_bboxes', 'gt_labels', 'gt_masks','gt_polygons', 'gt_offsets','gt_pitch_ratio'],meta_keys=['img_shape']),
   ]

train_ann_file=data_root + 'rotated_with_pitch/train/annotation.json'
test_ann_file=data_root + 'rotated_with_pitch/test/annotation.json'
train_img_prefix=data_root + 'rotated/train/images'
test_img_prefix=data_root + 'rotated/test/images'
data = dict(
    samples_per_gpu=2,# batch_size=num_gpu*samples_per_gpu
    workers_per_gpu=0,# num_workes=num_gpu*workers_per_gpu
    train=dict(
        type=dataset_type,
        ann_file=train_ann_file,# annotation json path + file name
        img_prefix=train_img_prefix,# dir of images
        bbox_type='building',
        mask_type='roof',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=test_ann_file,
        img_prefix=test_img_prefix,
        bbox_type='building',
        mask_type='roof',
        gt_footprint_csv_file="",
        pipeline=test_pipeline,
        anno_pipeline=test_anno_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=test_ann_file,
        img_prefix=test_img_prefix,
        # ann_file=train_ann_file[0],# 替换测试数据
        # img_prefix=img_prefix[0],
        # ann_file=data_root + 'coco/bonai_{}_trainval.json'.format('beijing'),
        # img_prefix=data_root + "trainval/images/",
        bbox_type='building',
        mask_type='roof',
        gt_footprint_csv_file="",
        pipeline=test_pipeline,
        anno_pipeline=test_anno_pipeline
    )
)
polygonize_post_process = dict(
    method='simple',
    data_level=0.5,
    # tolerance=[0.125, 2],
    tolerance=[2],
    seg_threshold=0.5,
    min_area=10
)
evaluation = dict(interval=1,# 每多少个train_epoch验证一次
                  metric=['bbox', 'segm','offset','polygon'],# bonai数据集可以计算的metric包括coco的metirc：'bbox', 'proposal', 'proposal_fast', 'segm'，在加上offset和angle
                  show=True,# 验证过程中是否进行可视化
                  shwo_interval=200,#  验证过程中每shwo_interval张图片可视化一次
                  show_score_thr=0.5)# 验证过程中bbox的score超过thr才显示
checkpoint_config = dict(type='SaveBestCkptHook',save_best_metrices=['bbox_mAP'],interval=1,max_keep_ckpts=5)