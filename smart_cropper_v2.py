
import os
import numpy as np
import rasterio
from rasterio.windows import Window
import random
from collections import Counter
from skimage import measure
import cv2

from config import Regions, Classes


class SmartCropperV2:
    def __init__(self,
                 image_path,
                 mask_path,
                 region_map_path,  # 新增区域引导图路径
                 output_image_dir,
                 output_mask_dir,
                 output_region_map_dir,  # 输出区域引导图路径
                 output_unlabeled_dir,
                 patch_size=256,
                 stride=128,
                 classes=Classes,
                 Regions=Regions,
                 ignore_index=255,
                 nodata_val=0,
                 min_valid_ratio=0.05,
                 min_class_ratio=0.05,
                 dense_region_sample_ratio=0.8,
                 random_samples_per_dense_region=5):
        if classes is None:
            classes = [0, 1, 2, 3]
        self.Regions = Regions
        self.image_path = image_path
        self.mask_path = mask_path
        self.region_map_path = region_map_path  # 保存区域引导图路径
        self.output_image_dir = output_image_dir
        self.output_mask_dir = output_mask_dir
        self.output_region_map_dir = output_region_map_dir  # 输出区域引导图目录
        self.output_unlabeled_dir = output_unlabeled_dir
        self.patch_size = patch_size
        self.stride = stride
        self.classes = classes
        self.ignore_index = ignore_index
        self.nodata_val = nodata_val
        self.min_valid_ratio = min_valid_ratio
        self.min_class_ratio = min_class_ratio
        self.dense_region_sample_ratio = dense_region_sample_ratio
        self.random_samples_per_dense_region = random_samples_per_dense_region

        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        os.makedirs(output_region_map_dir, exist_ok=True)  # 创建区域引导图目录
        os.makedirs(output_unlabeled_dir, exist_ok=True)

    def analyze_label_distribution(self, mask_patch):
        flat = mask_patch.flatten()
        total = np.sum(np.isin(flat, self.classes))
        counter = Counter(flat)
        return {cls: counter[cls] / total if total > 0 else 0 for cls in self.classes}

    def is_valid_patch(self, mask_patch):
        proportions = self.analyze_label_distribution(mask_patch)
        valid_classes = sum(1 for v in proportions.values() if v > self.min_class_ratio)
        return valid_classes >= 2

    def _save_patch(self, patch, src, window, path):
        profile = src.profile.copy()
        profile.update({
            'count': patch.shape[0],
            'height': self.patch_size,
            'width': self.patch_size,
            'transform': rasterio.windows.transform(window, src.transform)
        })
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(patch)

    def _save_mask(self, mask_patch, src, window, path):
        profile = src.profile.copy()
        profile.update({
            'count': 1,
            'height': self.patch_size,
            'width': self.patch_size,
            'transform': rasterio.windows.transform(window, src.transform)
        })
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(mask_patch, 1)
        # 新增：保存区域引导图

    def _save_region_map(self, region_map_patch, src, window, path):
        profile = src.profile.copy()
        profile.update({
            'count': 1,
            'height': self.patch_size,
            'width': self.patch_size,
            'transform': rasterio.windows.transform(window, src.transform)
        })
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(region_map_patch, 1)

    def detect_dense_label_regions(self, mask, min_region_area=2000):
        binary_mask = np.isin(mask, self.classes).astype(np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        labeled = measure.label(binary_mask, connectivity=2)
        props = measure.regionprops(labeled)
        regions = []
        for prop in props:
            if prop.area >= min_region_area:
                minr, minc, maxr, maxc = prop.bbox
                regions.append((minc, minr, maxc, maxr))
        return regions

    def extract_patches(self):
        count = 0 # 计数器，统计生成的 patch 数量

        # 打开遥感影像和对应标签
        with rasterio.open(self.image_path) as src_img, rasterio.open(self.mask_path) as src_mask,rasterio.open(self.region_map_path) as src_region:  # 打开区域引导图:
            img_width = src_img.width
            img_height = src_img.height

            # 读取完整标签图，用于检测“标签密集区域”
            full_mask = src_mask.read(1)
            dense_regions = self.detect_dense_label_regions(full_mask)

            # ========= 第一阶段：滑动窗口裁剪全图 =========
            for top in range(0, img_height - self.patch_size + 1, self.stride):
                for left in range(0, img_width - self.patch_size + 1, self.stride):
                    # 构建裁剪窗口
                    window = Window(left, top, self.patch_size, self.patch_size)

                    # 读取图像和标签 patch
                    img_patch = src_img.read([1, 2, 3], window=window)
                    mask_patch = src_mask.read(1, window=window)
                    # 读取对应区域引导图 patch
                    region_map_patch = src_region.read(1, window=window)

                    # ① 判断图像 patch 是否为无效区域（全为0或NaN）
                    if np.all((img_patch == self.nodata_val) | np.isnan(img_patch)):
                        continue
                    # ② 判断图像 patch 中有效像素比例是否低于阈值
                    valid_pixel_ratio = np.mean((img_patch != self.nodata_val) & (~np.isnan(img_patch)))
                    if valid_pixel_ratio < self.min_valid_ratio:
                        continue

                    # 如果区域引导图的值是 nodata，而图像有有效像素值，则将区域引导图的值替换为 1
                    region_map_patch[~np.isin(region_map_patch, self.Regions)] = 1
                    # region_map_patch[region_map_patch == self.nodata_val] = 1

                    # ③ 屏蔽无效标签值，设为 ignore_index（如255）
                    mask_patch[~np.isin(mask_patch, self.classes)] = self.ignore_index
                    # ④ 判断标签 patch 是否完全无效
                    if np.all((mask_patch == self.ignore_index) | np.isnan(mask_patch)):
                        path = os.path.join(self.output_unlabeled_dir, f"patch_{count:04d}.tif")
                        # 保存为未标注图像（图像有值但标签无效）
                        self._save_patch(img_patch, src_img, window, path)
                        region_map_path = os.path.join(self.output_region_map_dir, f"patch_{count:04d}.tif")  # 保存区域引导图
                        self._save_region_map(region_map_patch, src_region, window, region_map_path)  # 保存区域引导图
                        count += 1
                        continue

                    # ⑤ 判断标签 patch 是否有效（占比太低或单一类别时丢弃）
                    mask_valid_ratio = np.mean(mask_patch != self.ignore_index)
                    if mask_valid_ratio < self.min_valid_ratio or not self.is_valid_patch(mask_patch):
                        path = os.path.join(self.output_unlabeled_dir, f"patch_{count:04d}.tif")
                        self._save_patch(img_patch, src_img, window, path)
                        region_map_path = os.path.join(self.output_region_map_dir, f"patch_{count:04d}.tif")  # 保存区域引导图
                        self._save_region_map(region_map_patch, src_region, window, region_map_path)  # 保存区域引导图
                        count += 1
                        continue
                    # ⑥ 图像和标签都有效，保存
                    img_path = os.path.join(self.output_image_dir, f"patch_{count:04d}.tif")
                    mask_path = os.path.join(self.output_mask_dir, f"patch_{count:04d}.tif")
                    region_map_path = os.path.join(self.output_region_map_dir,f"patch_{count:04d}.tif")  # 保存区域引导图
                    self._save_patch(img_patch, src_img, window, img_path)
                    self._save_mask(mask_patch, src_mask, window, mask_path)
                    self._save_region_map(region_map_patch, src_region, window, region_map_path)  # 保存区域引导图
                    count += 1

            # ========= 第二阶段：随机裁剪标签密集区域 =========
            for region in dense_regions:
                left0, top0, right0, bottom0 = region
                area = (right0 - left0) * (bottom0 - top0)

                # ✅ 根据面积估计可容纳多少 patch（例如 1 patch = patch_size² = 65536）
                estimated_patches = area // (self.patch_size * self.patch_size)

                # ✅ 加权采样数量（调整权重或加 bias）
                num_samples = min(max(1, estimated_patches), 10)  # 最少 1，最多 10

                for _ in range(num_samples):
                    # 在该区域内随机采样一个 patch 起点
                    rand_left = random.randint(left0, max(left0, right0 - self.patch_size))
                    rand_top = random.randint(top0, max(top0, bottom0 - self.patch_size))
                    window = Window(rand_left, rand_top, self.patch_size, self.patch_size)
                    img_patch = src_img.read([1, 2, 3], window=window)
                    mask_patch = src_mask.read(1, window=window)
                    mask_patch[~np.isin(mask_patch, self.classes)] = self.ignore_index
                    # 排除无效或低质量 patch
                    if np.all((mask_patch == self.ignore_index) | np.isnan(mask_patch)):
                        continue
                    if not self.is_valid_patch(mask_patch):
                        continue
                    # 保存有效图像和标签
                    # 如果区域引导图的值是 nodata，而图像有有效像素值，则将区域引导图的值替换为 1
                    region_map_patch[~np.isin(region_map_patch, self.Regions)] = 1

                    img_path = os.path.join(self.output_image_dir, f"patch_{count:04d}.tif")
                    mask_path = os.path.join(self.output_mask_dir, f"patch_{count:04d}.tif")
                    region_map_path = os.path.join(self.output_region_map_dir, f"patch_{count:04d}.tif")
                    self._save_patch(img_patch, src_img, window, img_path)
                    self._save_mask(mask_patch, src_mask, window, mask_path)
                    self._save_region_map(region_map_patch, src_region, window, region_map_path)
                    count += 1
        print(f"✅ 智能裁剪完成，共生成 patch 数量：{count}")
