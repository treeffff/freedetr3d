import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description="Draw histogram of 3D coordinates from KITTI labels")
    parser.add_argument("--label_dir", type=str, default="data/dair_kitti/training/label_2", help="Directory containing KITTI label_2 txt files")
    parser.add_argument("--out_dir", type=str, default="./hist_output", help="Directory to save histogram images")
    parser.add_argument("--classes", type=str, nargs="+", default=["Car", "Van", "Truck", "Bus"], help="Target object classes")
    return parser.parse_args()


def parse_kitti_labels(label_path, target_classes):
    coords = []
    with open(label_path, 'r') as f:
        for line in f:
            elems = line.strip().split()
            if len(elems) < 15:
                continue
            obj_class = elems[0]
            if obj_class not in target_classes:
                continue
            x = float(elems[11])
            y = float(elems[12])
            z = float(elems[13])
            if z > 90:
                continue
            coords.append((x, y, z))
    return coords


def draw_histograms(all_coords, out_dir):
    coords = np.array(all_coords)
    if coords.shape[0] == 0:
        print("No coordinates found to plot.")
        return

    x_vals, y_vals, z_vals = coords[:, 0], coords[:, 1], coords[:, 2]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].hist(x_vals, bins=50, color='r')
    axs[0].set_title('X Coordinate Histogram')
    axs[0].set_xlabel('X (meters)')
    axs[0].set_ylabel('Count')

    axs[1].hist(y_vals, bins=50, color='g')
    axs[1].set_title('Y Coordinate Histogram')
    axs[1].set_xlabel('Y (meters)')
    axs[1].set_ylabel('Count')

    axs[2].hist(z_vals, bins=50, color='b')
    axs[2].set_title('Z Coordinate Histogram')
    axs[2].set_xlabel('Z (meters)')
    axs[2].set_ylabel('Count')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lhw.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved histogram to {out_path}")


if __name__ == "__main__":
    args = get_args()
    all_coords = []
    for file_name in os.listdir(args.label_dir):
        if not file_name.endswith(".txt"):
            continue
        label_path = os.path.join(args.label_dir, file_name)
        coords = parse_kitti_labels(label_path, args.classes)
        all_coords.extend(coords)

    draw_histograms(all_coords, args.out_dir)
