import os
from torch.utils.data import Dataset
import rasterio
import numpy as np
import torch

from config import num_classes, IGNORE_INDEX, num_region_types
from utils import get_region_id_from_map


# ------------------------------------------------------------
#有标签数据集加载
class CroplandDataset_1(Dataset):
    def __init__(self, image_dir, label_dir,  region_map_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.region_map_dir = region_map_dir  # 新增：区域引导图路径
        self.transform = transform
        self.image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif') or f.endswith('.png')])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name)
        region_map_path = os.path.join(self.region_map_dir, image_name)  # 获取对应的区域引导图路径

        #读取图像
        with rasterio.open(image_path) as src:
         image = src.read([1, 2, 3])  # RGB波段
         image = np.transpose(image, (1, 2, 0))  # C,H,W -> H,W,C

        image = image.astype(np.float32)
        image = np.nan_to_num(image, nan=0.0)
        # image = image.astype(np.float32) / 10000.0  # 归一化

        #读取标签数据
        with rasterio.open(label_path) as src:
            label = src.read(1)                 # 只取单通道
            label = label.astype(np.uint8)          # 转 0/1/255
        # 保证 label 的范围在 [0, num_classes - 1] 之间
        label = np.where(np.isnan(label), IGNORE_INDEX, label)
        label[label >= num_classes] = IGNORE_INDEX  # 忽略值
        label[label < 0] = IGNORE_INDEX  # 忽略值

        # 读取区域引导图（region map）
        with rasterio.open(region_map_path) as src:
            region_map = src.read(1)  # 读取区域引导图数据
        region_map = np.nan_to_num(region_map, nan=1)  # NaN -> 1，避免使用 IGNORE_INDEX
        region_map = np.clip(region_map, 0, num_region_types - 1)  # 确保 region_id 在合法范围内
        region_map[region_map >= num_region_types] = 1  # 忽略值
        region_map[region_map < 0] = 1  # 忽略值


        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = torch.from_numpy(label).long()
        # 获取每个 patch 的区域 ID
        region_id = get_region_id_from_map(region_map)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            region_map = self.transform(region_map)
        return image, label, region_map,region_id # 返回图像、标签、区域引导图和区域标识

# ---推理时加载数据集---
class CroplandInferenceDataset(Dataset):
    def __init__(self, image_dir, region_map_dir, transform=None):
        self.image_dir = image_dir
        self.region_map_dir = region_map_dir  # 区域引导图路径
        self.transform = transform
        self.image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif') or f.endswith('.png')])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        region_map_path = os.path.join(self.region_map_dir, image_name)  # 获取对应的区域引导图路径

        # 读取图像
        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3])  # RGB波段
            image = np.transpose(image, (1, 2, 0))  # C,H,W -> H,W,C

        image = image.astype(np.float32)
        image = np.nan_to_num(image, nan=0.0)

        # 读取区域引导图（region map）
        with rasterio.open(region_map_path) as src:
            region_map = src.read(1)  # 读取区域引导图数据
        region_map = np.nan_to_num(region_map, nan=1)  # NaN -> 1，避免使用 IGNORE_INDEX
        region_map = np.clip(region_map, 0, num_region_types - 1)  # 确保 region_id 在合法范围内
        region_map[region_map >= num_region_types] = 1  # 忽略值
        region_map[region_map < 0] = 1  # 忽略值

        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # 获取每个图像的区域 ID
        region_id = get_region_id_from_map(region_map)

        if self.transform:
            image = self.transform(image)
            # region_map = self.transform(region_map)

        return image, region_id,image_name  # 返回图像和区域 ID


# ------------------------------------------------------------
#无标签数据集加载
class CroplandDatasetUnlabeled(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        初始化无标签数据集
        :param image_dir: 图像文件夹路径
        :param transform: 数据增强或预处理操作
        """
        self.image_dir = image_dir
        self.transform = transform
        # 获取目录下所有图像文件（支持 tif 和 png）
        self.image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif') or f.endswith('.png')])

    def __len__(self):
        """
        返回数据集中的样本数
        """
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        :param idx: 数据索引
        :return: 返回图像数据
        """
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # 使用 rasterio 读取图像（支持 GeoTIFF 格式）
        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3])  # 读取 RGB 波段数据
            image = np.transpose(image, (1, 2, 0))  # 转换形状，从 (C, H, W) -> (H, W, C)

        image = image.astype(np.float32)
        image = np.nan_to_num(image, nan=0.0)
        # image = image.astype(np.float32) / 10000.0  # 对图像进行归一化，假设原始值在 [0, 10000] 范围内

        # 将 NumPy 数组转换为 PyTorch 张量，并调整通道顺序 (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # 变为 (C, H, W)

        # 如果有数据增强操作，则应用这些变换
        if self.transform:
            image = self.transform(image)

        return image  # 返回图像，不返回标签
    def get_image_name(self, idx):
        name = self.image_list[idx]
        name = os.path.splitext(name)[0]
        return name

    def get_image_idx(self, idx):
        name = self.image_list[idx]
        name = os.path.splitext(name)[0]
        # 分割并取最后一部分
        num = int(name.split("_")[-1])
        return num

    def read_image(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # 使用 rasterio 读取图像（支持 GeoTIFF 格式）
        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3])  # 读取 RGB 波段数据
            image = np.transpose(image, (1, 2, 0))  # 转换形状，从 (C, H, W) -> (H, W, C)

        image = image.astype(np.float32)
        image = np.nan_to_num(image, nan=0.0)
        # image = image.astype(np.float32) / 10000.0  # 对图像进行归一化，假设原始值在 [0, 10000] 范围内

        # 将 NumPy 数组转换为 PyTorch 张量，并调整通道顺序 (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # 变为 (C, H, W)


        return image  # 返回图像，不返回标签


# ------------------------------------------------------------
# 自定义一个混合数据集：真标签 + 稳定伪标签
class HybridDataset(torch.utils.data.Dataset):
    def __init__(self, sup_img, sup_lbl,region_map_dir,meta, unlabeled_imgs, transform=None):
        self.sup_ds    = CroplandDataset_1(sup_img, sup_lbl,transform)
        self.meta      = meta
        self.ul_imgs   = unlabeled_imgs
        self.region_map_dir = region_map_dir
        self.transform = transform
        # 获取真实标签和伪标签的数量
        self.num_real_labels = len(self.sup_ds)  # 真实标签数量
        self.num_undo_labels = len(self.meta)  # 伪标签数量
        # 计算扩增因子
        self.num_extra_real_labels = self.num_undo_labels - self.num_real_labels  # 需要增加的数量
        self.real_label_indices = list(range(self.num_real_labels))  # 真实标签的索引列表
        # 扩增真实标签的索引，直到数量与伪标签一致
        if self.num_extra_real_labels > 0:
            self.real_label_indices += self.real_label_indices * (self.num_extra_real_labels // self.num_real_labels)
            self.real_label_indices += self.real_label_indices[:(self.num_extra_real_labels % self.num_real_labels)]

    def __len__(self):
        return len(self.sup_ds) + len(self.meta)

    def __getitem__(self, idx):
        # ① 如果索引小于真实标签数据集的长度，或者是扩增后的真实标签
        if idx < len(self.sup_ds):
            return self.sup_ds[idx]   # (img, label)
        elif idx < len(self.real_label_indices) + len(self.sup_ds):
            real_idx = self.real_label_indices[idx - len(self.sup_ds)]  # 从扩增后的真实标签索引中获取
            # 使用原始真实标签数据集中的图像和标签
            return self.sup_ds[real_idx]

        # 后面是伪标签
        m   = self.meta[idx - len(self.sup_ds)]
        # 使用 rasterio 读取图像（支持 GeoTIFF 格式）
        img_path = os.path.join(self.ul_imgs,m["img_idx"])
        with rasterio.open(img_path) as src:
            img = src.read([1, 2, 3])  # 读取 RGB 波段数据
            img = np.transpose(img, (1, 2, 0))  # 转换形状，从 (C, H, W) -> (H, W, C)

        img = img.astype(np.float32)
        img = np.nan_to_num(img, nan=0.0)
        # image = image.astype(np.float32) / 10000.0  # 对图像进行归一化，假设原始值在 [0, 10000] 范围内

        # 将 NumPy 数组转换为 PyTorch 张量，并调整通道顺序 (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # 变为 (C, H, W)

        #读取生成的标签
        lbl = np.load(m["pseudo"])

        # 读取区域引导图（region map）
        region_map_path = os.path.join(self.region_map_dir,m["img_idx"])
        with rasterio.open(region_map_path) as src:
            region_map = src.read(1)  # 读取区域引导图数据
        region_map = np.nan_to_num(region_map, nan=1)  # NaN -> 1，避免使用 IGNORE_INDEX
        region_map = np.clip(region_map, 0, num_region_types - 1)  # 确保 region_id 在合法范围内
        region_map[region_map >= num_region_types] = 1  # 忽略值
        region_map[region_map < 0] = 1  # 忽略值

        if self.transform:
            img, lbl,region_map= self.transform(img, lbl,region_map)
        region_id = get_region_id_from_map(region_map)
        return img, torch.from_numpy(lbl).long(),region_map,region_id
