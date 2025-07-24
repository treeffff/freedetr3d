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
    parser.add_argument("--occ", type=int, default=256, help="Occupancy map size (in pixels)")

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


# lenth: 90m, width: 25m
# res: l/w  / 
def get_3Dbox_to_2Dbox(label_path, range_x, range_y, occ, out_dir1, out_dir2):

    if label_path[-3:] != "txt":
        return

    FVView = np.zeros((occ, occ))
    BEVView = np.zeros((occ, occ))

    labels = open(label_path).read()
    labels = labels.split("\n")
    img1 = Image.fromarray(FVView)
    img2 = Image.fromarray(BEVView)
    for label in labels:
        if label == "":
            continue

        elems = label.split()
        if elems[0] in ['Car', 'Van', 'Bus', 'Truck']:
            
            center_x = float(elems[11]) / range_x * occ + occ / 2
            center_y = float(elems[12]) / range_y * occ + occ * (4 / 5)
            center_z = occ - float(elems[13]) / range_x * occ

            orient = -1 * float(elems[14])

            obj_h = int(float(elems[8]) / range_y * occ)
            obj_l = int(float(elems[10]) / range_x * occ)
            obj_w = int(float(elems[9]) / range_x * occ)

            rectangle = get_rect(0, 0, obj_l, obj_w, orient)
            ori_l = max(rectangle[:, 0]) - min(rectangle[:, 0])

            rectangle_bev =  get_rect(center_x, center_z, obj_l, obj_w, orient)
            rectangle_fv = get_rect(center_x, center_y, ori_l, obj_h, 0)

            draw_bev = ImageDraw.Draw(img1)
            draw_bev.polygon([tuple(p) for p in rectangle_bev], fill=255)

            draw_fv = ImageDraw.Draw(img2)
            draw_fv.polygon([tuple(p) for p in rectangle_fv], fill=255)

    img1 = img1.convert('L')
    img2 = img2.convert('L')
    file_path1 = os.path.join(
        out_dir1,
        os.path.basename(label_path)[
            :-3] + "png")
    file_path2 = os.path.join(
        out_dir2,
        os.path.basename(label_path)[
            :-3] + "png")
    img1.save(file_path1)
    print("Saved file at %s" % file_path1)
    img2.save(file_path2)
    print("Saved file at %s" % file_path2)
    

if __name__ == "__main__":
    args = get_args()
    args.out_dir1 = "data/dair_kitti/training/bev_seg_2"
    args.out_dir2 = "data/dair_kitti/training/fv_seg_2"
    args.base_path = "data/dair_kitti/training/label_2"

    if not os.path.exists(args.out_dir1):
        os.makedirs(args.out_dir1)
    if not os.path.exists(args.out_dir2):
        os.makedirs(args.out_dir2)

    for file_path in os.listdir(args.base_path):
        label_path = os.path.join(args.base_path, file_path)
        get_3Dbox_to_2Dbox(
            label_path,
            args.range_x,
            args.range_y,
            args.occ,
            args.out_dir1,
            args.out_dir2)

        