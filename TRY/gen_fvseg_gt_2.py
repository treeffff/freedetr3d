import argparse
import os
import pdb

from PIL import Image, ImageDraw

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="MonoLayout DataPreparation options")
    parser.add_argument("--dataset", type=str, default="dair_c_inf", help="dair_c_inf or dair_i")
    parser.add_argument("--range_x", type=int, default=90, help="Size of the rectangular grid in metric space (in m)")
    parser.add_argument("--range_y", type=int, default=25, help="Size of the rectangular grid in metric space (in m)")
    parser.add_argument("--occ_map_size", type=int, default=256, help="Occupancy map size (in pixels)")

    return parser.parse_args()


def get_rect(x, y, width, height, theta):
    rect = np.array([(-width / 2, -height / 2), (width / 2, -height / 2),
                     (width / 2, height / 2), (-width / 2, height / 2),
                     (-width / 2, -height / 2)])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset

    return transformed_rect


# lenth: 90m, width: 5m
# res: l/w  / 
def get_3Dbox_to_2Dbox(label_path, width, height, res_x, res_y, out_dir):

    if label_path[-3:] != "txt":
        return

    TopView = np.zeros((int(width / res_x), int(height / res_y)))
    labels = open(label_path).read()
    labels = labels.split("\n")
    img = Image.fromarray(TopView)
    for label in labels:
        if label == "":
            continue

        elems = label.split()
        if elems[0] in ['Car', 'Van', 'Bus', 'Truck']:
            center_x = int(float(elems[11]) / res_x + width / (2 * res_x))
            center_y = int(float(elems[12]) / res_y) + height / res_y * (4 / 5)
            # print (center_y, int(float(elems[8])), float(elems[12]), float(elems[8]))
            orient = -1 * float(elems[14])

            obj_h = int(float(elems[8]) / res_y)
            obj_l = int(float(elems[10]) / res_x)
            obj_w = int(float(elems[9]) / res_x)

            rectangle_bev = get_rect(0, 0, obj_l, obj_w, orient)
            ori_l = max(rectangle_bev[:, 0]) - min(rectangle_bev[:, 0])


            rectangle = get_rect(center_x, center_y, ori_l, obj_h, 0)
            draw = ImageDraw.Draw(img)
            draw.polygon([tuple(p) for p in rectangle], fill=255)

    img = img.convert('L')
    file_path = os.path.join(
        out_dir,
        os.path.basename(label_path)[
            :-3] + "png")
    img.save(file_path)
    print("Saved file at %s" % file_path)
    

if __name__ == "__main__":
    args = get_args()
    args.out_dir = "data/dair_kitti/training/bev_seg_4"
    args.base_path = "data/dair_kitti/training/label_2"

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    i = 0
    for file_path in os.listdir(args.base_path):
        label_path = os.path.join(args.base_path, file_path)
        get_3Dbox_to_2Dbox(
            label_path,
            args.range_x,
            args.range_y,
            args.range_x/float(args.occ_map_size),
            args.range_y/float(args.occ_map_size),
            args.out_dir)
        i += 1
        if i % 100 == 0:
            break