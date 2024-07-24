# encoding:utf-8
# model settings
model = dict(
    type='CrossfieldMultiScale2',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    crossfield_head=dict(
        type='UNetHead',
        n_channels=[2048,1024,512,256],
        n_classes=4,
        bilinear=False,
        loss_crossfield_align=dict(
            type='CrossfieldAlignOffNadirLoss',level_2_align=True),
        loss_crossfield_smooth=dict(
            type='CrossfieldSmoothLoss')
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='UNetRoIHead2',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='MultiScaleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_sizes=[112,56,28,14], sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='UNetMaskCrossfieldConstraintHead2',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            class_agnostic=False,
            crossfield_roi_align=dict(type='RoIAlign', output_sizes=[28,56,112,224], sampling_ratio=0,spatial_scale=256/1024),# 2*mask_roi_extractor
            loss_mask=dict(
                type='CrossEntropyLoss', loss_weight=1.0,use_mask=True),
            loss_roof_facade_horiontal=dict(
                type='HorizontalSegGradLoss', left_channel=1,right_channel=2),
            loss_roof_crossfield=dict(
                type='CrossfieldGradAlignLoss',
                pre_channel=0,level_2_align=True), # todo:加入edge相关元素
                # 256:crossfield_size type=RoIAlign:crossfield_grad_align_loss.py
        ),
        offset_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        offset_head=dict(
            type='OffsetHeadExpandFeature',
            expand_feature_num=4,
            share_expand_fc=True,
            rotations=[0, 90, 180, 270],
            num_fcs=2,
            fc_out_channels=1024,
            num_convs=10,
            loss_offset=dict(type='SmoothL1Loss', loss_weight=8*2.0)),
    ),
    loss_weights=dict(
        loss_crossfield_align=3,
        loss_crossfield_smooth=0.2,
        loss_rpn_cls=1,
        loss_rpn_bbox=1,
        loss_cls=1,
        loss_bbox=1,
        loss_mask=2,
        loss_roof_crossfield=0.2,
        loss_roof_facade_horiontal=0,
        loss_offset=0.5
    ),
)
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1,
            gpu_assign_thr=512),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=3000,
        nms_post=3000,
        max_num=3000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=True,
            ignore_iof_thr=-1,
            gpu_assign_thr=512),
        sampler=dict(
            type='RandomSampler',
            num=1024,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=[28,56,112,224],
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=3000,
        nms_post=3000,
        max_num=3000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='soft_nms', iou_threshold=0.5),
        max_per_img=500,# 此时是多类别分割，每个mask包含了三个通道，导致原来的2000个mask会占用很多操作时间和空间，因此在此处降低采用的目标数量
        mask_thr_binary=0.5))# 对于off_nadir多类mask数据返回logits会导致内存不够！
        # mask_thr_binary=-1))# -1输出的就不是二值的mask，而是模型响应大小，并且不经过rle紧凑化存储
