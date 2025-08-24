import json

import torch
from rasterio._io import Window
from torch import nn

import torch.nn.functional as F
from config import IGNORE_INDEX, Classes, initial_region_prior
import os
import rasterio
import numpy as np
from rasterio.windows import Window
import albumentations as A
from albumentations.pytorch import ToTensorV2

#从整幅图或 patch 中投票得到主 region_id
def get_region_id_from_map(region_map_patch: torch.Tensor) -> torch.Tensor:
    """
    输入:
        region_map_patch: (H, W)，区域引导图中的每个像素是一个 int 区域编号
    输出:
        region_id: int，主区域编号（该 patch 中区域数量最多的区域）
    """
    if isinstance(region_map_patch, np.ndarray):
        region_map_patch = torch.tensor(region_map_patch)

    region_ids, counts = torch.unique(region_map_patch, return_counts=True)
    region_id = region_ids[torch.argmax(counts)]  # 选择像素最多的区域编号
    return region_id

# ：soft label 构建（训练阶段使用）
# ✅ 第三步：soft label 构建（使用 learnable region_prior 参数）
def build_soft_region_label(region_map, learned_prior):
    # learned_prior: shape (num_region_types, num_classes)
    B, H, W = region_map.shape
    num_classes = learned_prior.shape[1]
    soft_label = torch.zeros((B, num_classes, H, W), device=region_map.device)
    for region_id in range(learned_prior.shape[0]):
        mask = (region_map == region_id).unsqueeze(1)  # (B,1,H,W)
        prob_tensor = learned_prior[region_id].view(1, -1, 1, 1)
        soft_label += mask.float() * prob_tensor
    return soft_label

# ✅ soft_cross_entropy 实现（带 logit 输入 + soft target）
def soft_cross_entropy(pred, soft_target,eps: float = 1e-8):
    soft_target = soft_target.clamp_min(0)  # 处理负值
    soft_target = soft_target / soft_target.sum(dim=1, keepdim=True).clamp_min(eps)  # 归一化

    log_prob = F.log_softmax(pred, dim=1)
    return -(soft_target * log_prob).sum(dim=1).mean()
# ------------------------------------------------------------
# MIOU计算
def miou(pred_a: torch.Tensor, pred_b: torch.Tensor, n_cls: int) -> float:
    """
    pred_a, pred_b: int64 [H,W] on CPU
    return: scalar float
    """
    ious = []
    for cls in range(n_cls):
        inter = torch.logical_and(pred_a == cls, pred_b == cls).sum()
        union = torch.logical_or (pred_a == cls, pred_b == cls).sum()
        if union > 0:
            ious.append( inter.float() / union.float() )
    if not ious:          # 图里没有任何类别像素
        return 0.0
    return torch.stack(ious).mean().item()

# ------------------------------------------------------------
# 数据增强操作
def get_simclr_augmentations(image_size=256, use_blur=True, use_cutout=True):
    """
    构建与 SimCLR 论文一致的数据增强组合。
    参数：
        image_size: 图像输出尺寸，如 256 或 512
        use_blur: 是否加入 Gaussian Blur（ImageNet 中强烈建议）
        use_cutout: 是否加入 Cutout 遮挡增强
    返回：
        Albumentations.Compose 对象，可用于 image+mask 同步增强
    """
    aug_list = [
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.08, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
        A.ToGray(p=0.2),
    ]

    if use_blur:
        aug_list.append(A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.5))

    if use_cutout:
        aug_list.append(A.CoarseDropout(
            max_holes=1,
            max_height=int(image_size * 0.4),
            max_width=int(image_size * 0.4),
            min_holes=1,
            fill_value=0,
            mask_fill_value=0,
            p=0.5
        ))

    aug_list.append(ToTensorV2())

    return A.Compose(aug_list)

# ------------------------------------------------------------
# 递归把 BatchNorm2d → GroupNorm

# bn_to_gn里面的选组数：返回 ≤ prefer 且能整除 C 的最大整数；若没有则返回 1
def _best_group(C: int, prefer: int = 32) -> int:
    for g in reversed(range(1, prefer + 1)):
        if C % g == 0:
            return g
    return 1

# 递归把 BatchNorm2d → GroupNorm
def bn_to_gn(module: nn.Module, prefer_groups: int = 32, eps: float = 1e-5):
    """
    Examples
    --------
    model = CTHNet(num_classes=2).to(device)
    bn_to_gn(model, prefer_groups=32)   # ★ 一行搞定
    """
    for name, child in module.named_children():

        # ⚑ 若是 BatchNorm2d → 换成 GroupNorm
        if isinstance(child, nn.BatchNorm2d):
            C = child.num_features
            G = _best_group(C, prefer_groups)
            gn = nn.GroupNorm(num_groups=G, num_channels=C,
                              eps=eps, affine=True)
            setattr(module, name, gn)

        # ⚑ 递归下降到下一层
        else:
            bn_to_gn(child, prefer_groups, eps)

# ------------------------------------------------------------
#图像裁剪(单个影像裁剪)
def extract_patches(
    image_path,
    output_dir,
    patch_size=256,
    stride=128,
    nodata_val=0,
    min_valid_ratio=0.2,
    prefix="patch_"):
    """
    从大遥感影像中裁剪小图像块，仅保存有效图像（非空像素足够多）。

    参数：
    - image_path: str，输入大图路径
    - output_dir: str，输出小图目录
    - patch_size: int，裁剪图块尺寸（默认为256）
    - stride: int，滑动窗口步长（默认为128）
    - nodata_val: int，无效像素的值（默认为0）
    - min_valid_ratio: float，有效像素比例阈值（默认为0.2）
    - prefix: str，输出文件名前缀

    返回：
    - count: int，成功生成的patch数量
    """
    os.makedirs(output_dir, exist_ok=True)
    positions = []# 在裁剪代码中增加记录位置的逻辑
    count = 0

    with rasterio.open(image_path) as src_img:
        img_width = src_img.width
        img_height = src_img.height

        for top in range(0, img_height - patch_size + 1, stride):
            for left in range(0, img_width - patch_size + 1, stride):
                window = Window(left, top, patch_size, patch_size)
                img_patch = src_img.read([1, 2, 3], window=window)

                # 判断是否为有效图像
                if np.all(img_patch == nodata_val):
                    continue
                valid_ratio = np.mean(img_patch != nodata_val)
                if valid_ratio < min_valid_ratio:
                    continue

                # 更新图像信息并保存
                img_profile = src_img.profile.copy()
                img_profile.update({
                    'height': patch_size,
                    'width': patch_size,
                    'transform': rasterio.windows.transform(window, src_img.transform)
                })

                img_patch_path = os.path.join(output_dir, f"{prefix}{count:04d}.tif")
                with rasterio.open(img_patch_path, 'w', **img_profile) as dst:
                    dst.write(img_patch)
                positions.append((count, top, left))
                count += 1
    with open(os.path.join(output_dir, "patch_positions.json"), "w") as f:
        json.dump(positions, f)
    print(f"\n✅ 共生成 {count} 个有效 patch 图像，保存于：{output_dir}")
    return count

#图像裁剪
def extract_patches_all(
    image_path,#处理的遥感影像
    mask_path,#处理的对应标签tif文件
    output_image_dir,#裁剪成的（512*12或256*256）遥感影像片段文件地址
    output_mask_dir,#对应的标签片段文件地址
    output_unlabeled_dir,#对应未打标签的文件地址
    patch_size=256,#对应的片段大小尺寸
    stride=128,#滑动
    NODATA_VAL=0,
    min_crop_pixels = 2,  # 至少2个耕地像素
    min_valid_ratio=0.05,# 至少2个耕地像素
    prefix="patch_"# 至少20%像素不是0，才算有效patch
):

    # ========== 路径设置 ==========
    # image_path = r"G:\dqb\shashiqu\22\22image\shashiqu_22_rgb.tif"
    # mask_path = r"G:\dqb\shashiqu\22\22label_mask\label22.tif"
    # output_image_dir = r"G:\dqb\shashiqu\22\train_image"
    # output_mask_dir = r"G:\dqb\shashiqu\22\train_label"

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_unlabeled_dir, exist_ok=True)

    # ========== 参数设置 ==========
    # patch_size = 256
    # stride = 128
    # NODATA_VAL = 0
    # min_crop_pixels = 2   # 至少2个耕地像素
    # min_valid_ratio = 0.2  # 至少20%像素不是0，才算有效patch

    # prefix = "2022_"  # ✅ 自定义前缀，可换成你希望的名称，如 "2021_", "shashi_", 等

    count = 0

    with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
        img_width = src_img.width
        img_height = src_img.height

        for top in range(0, img_height - patch_size + 1, stride):
            for left in range(0, img_width - patch_size + 1, stride):
                window = Window(left, top, patch_size, patch_size)

                img_patch = src_img.read([1, 2, 3], window=window)
                mask_patch = src_mask.read(1, window=window)

                # 判断遥感影像是否全为0
                if np.all((img_patch == NODATA_VAL) | np.isnan(img_patch)):
                    continue

                # 判断遥感影像有效像素比例是否足够
                valid_pixel_ratio = np.mean((img_patch != NODATA_VAL) & (~np.isnan(img_patch)))
                # valid_pixel_ratio = np.mean(img_patch != NODATA_VAL)
                if valid_pixel_ratio < min_valid_ratio:
                    continue
                mask_patch[~np.isin(mask_patch, Classes)] = IGNORE_INDEX  # 比如设为15
                # 判断标签影像是否全为nan,自动将标签的NODATA_VAL设为15了
                if np.all((mask_patch == IGNORE_INDEX) | np.isnan(mask_patch)):
                    # ========== 保存图像 ==========
                    img_profile = src_img.profile.copy()
                    img_profile.update({
                        'count': 3,  # ✅ 设置为3个波段
                        'height': patch_size,
                        'width': patch_size,
                        'transform': rasterio.windows.transform(window, src_img.transform)
                    })

                    img_patch_path = os.path.join(output_unlabeled_dir, f"{prefix}patch_{count:04d}.tif")
                    with rasterio.open(img_patch_path, 'w', **img_profile) as dst:
                        dst.write(img_patch)
                    count += 1

                    continue

                # 判断标签影像有效像素比例是否足够
                valid_pixel_ratio = np.mean(mask_patch != IGNORE_INDEX)
                # valid_pixel_ratio = np.mean(np.all((mask_patch == IGNORE_INDEX)))
                if valid_pixel_ratio < min_valid_ratio:
                    # ========== 保存图像 ==========
                    img_profile = src_img.profile.copy()
                    img_profile.update({
                        'count': 3,  # ✅ 设置为3个波段
                        'height': patch_size,
                        'width': patch_size,
                        'transform': rasterio.windows.transform(window, src_img.transform)
                    })

                    img_patch_path = os.path.join(output_unlabeled_dir, f"{prefix}patch_{count:04d}.tif")
                    with rasterio.open(img_patch_path, 'w', **img_profile) as dst:
                        dst.write(img_patch)
                    count += 1

                    continue

                # ========== 保存图像 ==========
                img_profile = src_img.profile.copy()
                img_profile.update({
                    'count': 3,  # ✅ 设置为3个波段
                    'height': patch_size,
                    'width': patch_size,
                    'transform': rasterio.windows.transform(window, src_img.transform)
                })

                img_patch_path = os.path.join(output_image_dir, f"{prefix}patch_{count:04d}.tif")
                with rasterio.open(img_patch_path, 'w', **img_profile) as dst:
                    dst.write(img_patch)

                # ========== 保存标签 ==========
                mask_profile = src_mask.profile.copy()
                mask_profile.update({
                    'count': 1,  # ✅ 设置为3个波段
                    'height': patch_size,
                    'width': patch_size,
                    'transform': rasterio.windows.transform(window, src_mask.transform)
                })

                mask_patch_path = os.path.join(output_mask_dir, f"{prefix}patch_{count:04d}.tif")
                with rasterio.open(mask_patch_path, 'w', **mask_profile) as dst:
                    dst.write(mask_patch, 1)

                count += 1
    print(f"\n✅ 共生成 {count} 个有效 patch 图像和标签，满足像素有效比例 ≥ {min_valid_ratio*100:.0f}% 且标签中耕地像素 ≥ {min_crop_pixels}")

#加载对应模型并进行推理，并将结果保存到指定文件夹
#将矩阵数据转为tif文件并保存到指定文件夹
def save_prediction_tif(pred_array, reference_tif_path, save_path, dtype='uint8'):
    with rasterio.open(reference_tif_path) as src:
        profile = src.profile.copy()
        profile.update({
            'count': 1,
            'dtype': dtype,
            'nodata': 255,
            'driver': 'GTiff'
        })
        with rasterio.open(save_path, 'w', **profile) as dst:
            dst.write(pred_array.astype(dtype), 1)


