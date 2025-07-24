import json
import matplotlib.pyplot as plt

# 替换为你的日志文件路径
log_path = '/home/treefan/projects/detr3d/work_dirs/detr_free/20250630_200215.log.json'  # 如 'work_dirs/exp1/20240629_133000.log.json'

loss_list = []

# 读取日志
with open(log_path, 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        if data.get("mode") == "train" and "loss" in data:
            loss_list.append(data["loss"])

# 横轴：记录次数（0, 1, 2, ..., N）
x = list(range(len(loss_list)))

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(x, loss_list, label='Loss')
plt.xlabel('Log Record Index')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()