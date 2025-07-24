import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version



# 读取配置文件
cfg = Config.fromfile('/home/treefan/projects/detr3d/projects/configs/detr3d/detr3d_res101_gridmask.py')  # 这里换成你的配置文件路径


if hasattr(cfg, 'plugin'):
    if cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)
        else:
            # import dir is the dirpath for the config file
            _module_dir = os.path.dirname(args.config)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)

# 构建数据集
dataset = build_dataset(cfg.data.train)

# 打印一些信息
print(f"数据集长度：{len(dataset)}")

# 取一条数据测试
sample = dataset[0]
print("样本数据 keys：", sample.keys())
print("图像 shape：", sample['img'].data.shape)
# # print("gt_bboxes_3d：", sample['gt_bboxes_3d'].data.shape)
# print("gt_labels_3d：", sample['gt_labels_3d'].data.shape)
# print("img_metas：", sample['img_metas'])
# print("img_metas keys：", sample['points'].data.keys())

