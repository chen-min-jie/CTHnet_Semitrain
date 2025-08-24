# config.py - 全局路径与参数配置

import os

# 数据路径
supervised_img = "C:/Users/Administrator/Desktop/20250720/shashi_image"#train_image
supervised_lbl = "C:/Users/Administrator/Desktop/20250720/shashi_label"#train_label
region_map_dir = "C:/Users/Administrator/Desktop/20250720/region_map"

val_img = "G:/shashiqu/all_years/val_image"
val_lbl = "G:/shashiqu/all_years/val_label"

test_img = 'G:/shashiqu/all_years/test_image'
test_lbl = 'G:/shashiqu/all_years/test_label'

unlabeled_imgs = 'C:/Users/Administrator/Desktop/20250720/unlabeled_2023'
pseudo_lbl = 'C:/Users/Administrator/Desktop/20250720/20250720_semi_train/test1/pseudo_lbl'

# 教师模型保存路径
checkpoint_dir = 'E:/test/chenmin20250723'

#节点模型保存路径
checkpoint_CT  = 'E:/test/chenmin20250723/CT'
#学生模型保存的路径
checkpoint_Student = 'E:/test/chenmin20250723/Student'
# 模型结构参数
num_classes = 4
num_region_types = 5 #0，1，2，3，4
Regions = [0,2,3,4]
Classes = [0,1,2,3]
image_size = (256, 256)

# 训练参数
batch_size = 5
#有监督分类的迭代epoch数
epochs_supervised = 60
#半监督学生模型的迭代数
epochs_semi = 21

learning_rate = 1e-5
lambda_soft = 0.1
# 语义分割常用的无效像素值
IGNORE_INDEX = 255
# 推理参数
confidence_threshold = 0.9

initial_region_prior = [
    [0.7, 0.2, 0.05, 0.05],  # 水体：主要是水域（类别 0），少量为耕地（类别 1）或其他
    [0.1, 0.7, 0.15, 0.05],  # 耕地：主要是耕地（类别 1），少量为水域或其他
    [0.1, 0.05, 0.8, 0.05],  # 居民区：主要是居民区（类别 2），少量为耕地或山地
    [0.05, 0.1, 0.05, 0.8],  # 山地：主要是山地（类别 3），少量为水域或耕地
    [0.3, 0.4, 0.2, 0.1]     # 耕地多一点的山地：偏向耕地（类别 1）和山地（类别 3）
]


