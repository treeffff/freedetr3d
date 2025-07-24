import os
import json
import uuid

import os
import json
import uuid

def kitti_category_id(name):
    mapping = {
        'car': 0,
        'pedestrian': 1,
        'cyclist': 2
    }
    return mapping.get(name, -1)

def get_default_camera_info():
    return {
        "cam2ego_rotation": [
            0.4998015430569128,
            -0.5030316162024876,
            0.4997798114386805,
            -0.49737083824542755
        ],
        "cam2ego_translation": [
            1.70079118954,
            0.0159456324149,
            1.51095763913
        ],
        "ego2global_rotation": [
            0.4388360312827414,
            0.007251556779002451,
            0.003989370905325335,
            -0.898529040984249
        ],
        "ego2global_translation": [
            710.4369174164464,
            1794.8997834692777,
            0.0
        ],
        "cam_intrinsic": [
            [1266.417203046554, 0.0, 816.2670197447984],
            [0.0, 1266.417203046554, 491.50706579294757],
            [0.0, 0.0, 1.0]
        ],
        "lidar2img": [
            [ 0. , -1. ,  0. ,  1.2],
            [ 0. ,  0. , -1. ,  0.8],
            [ 1. ,  0. ,  0. ,  2.5],
            [ 0. ,  0. ,  0. ,  1. ]
        ],
        "width": 1920,
        "height": 1080,
    }


import os
import json


def convert_kitti_to_json(root_dir, output_path):
    annos = []
    images = []
    label_dir = os.path.join(root_dir, 'training', 'label_2')
    global_id = 0


    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        image_id = os.path.splitext(label_file)[0]

        file_name = f"training/image_2/{image_id}.jpg"
        label_path = os.path.join(label_dir, label_file)
        bev_seg = f"training/bev_seg/{image_id}.png"
        fv_seg = f"training/fv_seg/{image_id}.png"
        camera_info = get_default_camera_info()

        image_info = {
            "file_name": file_name,
            "bev_seg": bev_seg,
            "fv_seg": fv_seg,
            "id": image_id,  # 使用 image_id 可追溯
            "token": 0,
            "cam2ego_rotation": camera_info["cam2ego_rotation"],
            "cam2ego_translation": camera_info["cam2ego_translation"],
            "ego2global_rotation": camera_info["ego2global_rotation"],
            "ego2global_translation": camera_info["ego2global_translation"],
            "cam_intrinsic": camera_info["cam_intrinsic"],
            "lidar2img": camera_info["lidar2img"],
            "width": camera_info["width"],
            "height": camera_info["height"]
        }

        if global_id % 20 == 19:
            break
            print(f"Processing {global_id} images...")
        # 3. 遍历该图像对应的标注
        cnt = 0
        with open(label_path, 'r') as f:
            for line in f:
                fields = line.strip().split(' ')
                if len(fields) < 15:
                    continue

                name = fields[0]
                if name not in ['Car', 'Truck', 'Bus']:
                    continue
                name = 'car'
                bbox = list(map(float, fields[4:8]))  # 2D bbox
                h, l, w = map(float, fields[8:11])   # 3D尺寸
                x, y, z = map(float, fields[11:14])  # 3D中心点

                # point_cloud_range = [-45.0, 0.0, -5.0, 45.0, 90.0, 5.0]
                # 检查 3D 中心点是否在有效范围内
                if not (-45.0 <= x <= 45.0 and 0.0 <= z <= 90.0 and -5.0 <= y <= 5.0):
                    continue

                ry = float(fields[14])              # 旋转

                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                cx = bbox[0] + width / 2
                cy = bbox[1] + height / 2

                ann = {
                    "file_name": file_name,
                    "image_id": image_id,
                    "area": area,
                    "category_name": name,
                    "category_id": kitti_category_id(name),
                    "bbox": [bbox[0], bbox[1], width, height],
                    "iscrowd": 0,
                    "bbox_cam3d": [x, y, z, l, h, w, ry],
                    "velo_cam3d": [0.0, 0.0],
                    "center2d": [cx, cy, z],
                    "attribute_name": "unknown",
                    "attribute_id": -1,
                    "segmentation": [],
                    "id": global_id
                }
                global_id += 1
                annos.append(ann)
                cnt += 1
        if cnt == 0:
            print(file_name)
            continue
        images.append(image_info)
    output_json = {
        "annotations": annos,
        "images": images,
        "categories": [
            {"id": 0, "name": "car"},
            {"id": 1, "name": "pedestrian"},
            {"id": 2, "name": "cyclist"}
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(output_json, f, indent=2)

convert_kitti_to_json(root_dir='data/dair_kitti', output_path='data/dair_kitti/ann.json')