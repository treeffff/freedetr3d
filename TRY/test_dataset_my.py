input_modality = dict(use_lidar=False, use_camera=True)
dataset_type = 'NuScenesMonoDataset'
data_root = 'data/dair_kitti/'
class_names = ['car']
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=False,
        with_label=False,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=False),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes_3d', 'gt_labels_3d', 'bev_seg', 'fv_seg'
        ]),
]


test_pipeline = [
    dict(type='LoadImageFromFileMono3D', to_float32=True),  # 单目加载
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),  # 统一大小
    dict(type='Normalize', **img_norm_cfg),  # 归一化
    dict(type='Pad', size_divisor=32),  # padding保证尺寸是32的倍数
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1024, 1024),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ]
    )
]


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        box_type_3d='Camera'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'))


from mmdet3d.datasets import build_dataset
dataset = build_dataset(data['train'])
# for i in range(50):
#     print(dataset[i]['gt_labels_3d'].data)
# sample = dataset[2]
# print(dataset)
# print(sample['gt_labels_3d'])
# print(sample['gt_bboxes_3d'].data.tensor.shape)
# print(sample['gt_bboxes_3d'].data.tensor)
# print(sample.keys())
# # print(sample['bev_seg'].shape)
# # print(type(sample['bev_seg']))
# import numpy as np
# print(sample['img_metas'])
# # np.savetxt('bev_seg.txt', sample['bev_seg'].data.numpy(), fmt='%d')
# print(len(dataset))
print(dataset[19]['gt_labels_3d'].data)
print(dataset[19]['img_metas'].data)
print(dataset[19]['gt_bboxes_3d'].data)