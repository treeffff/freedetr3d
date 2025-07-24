path = "/home/treefan/projects/detr3d/data/dair_kitti/kitti_infos_trainval.pkl"
# 读取pkl文件并打印
import pickle
with open(path, 'rb') as f:
    data = pickle.load(f)
    print(len(data))
