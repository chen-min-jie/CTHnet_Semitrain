# -*- coding: utf-8 -*-
"""
按需求实现：
1) 读取单幅遥感影像及其对应区域引导图；以 512×512 裁剪（跳过全 0/NaN 的无效图像 patch），
   若该 patch 的区域引导图为 NoData，则替换为 1；将裁剪结果分别保存到 infer_image / infer_region。
2) 复用 infer_test 思路进行推理（调用 label_and_save_with_model），对裁剪后的 patch 推理并输出到 pred 目录。
3) 将推理后的 patch 按原位拼接成整体影像 infer_jimen.tif 和 infer_jimen2.tif，并将除值==1 外的像元全部置 0（得到二值图）。

使用准备：
- 需要安装 rasterio、numpy、torch 等依赖。
- 需要可用的 infer_test.py（其中包含 label_and_save_with_model 函数）、CTHNet 与 config.num_classes。

"""
import os
import json
import glob
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.crs import CRS

# ==== 模型相关（按 infer_test.py 的思路） ====
from infer_test import label_and_save_with_model  # noqa: F401
from config import num_classes, Regions  # noqa: F401


# ============ 路径与参数（可按需修改） ============
BASE_DIR = r"G:\20250820\2023"
IMAGE_PATH = r"E:\test\2023_jimmen_data\jinmen\jinmen_merged2023.tif"       # 输入影像（按用户描述示例命名）
REGION_PATH = r"G:\20250818\label_total\label_region1.tif"      # 区域引导图

PATCH_SIZE = 512
MODEL_PATH = r"G:\20250807\Teacher_train\model_epoch_120.pth"  # 替换为真实模型

# 裁剪输出
INFER_IMAGE_DIR = os.path.join(BASE_DIR, "infer_image")
INFER_REGION_DIR = os.path.join(BASE_DIR, "infer_region")
META_JSON = os.path.join(BASE_DIR, "grid_meta.json")

# 推理输出（patch 级结果）
PRED_DIR = os.path.join(BASE_DIR, "pred")

# 拼接输出（整幅影像）
OUT_TIF_1 = os.path.join(BASE_DIR, "infer_jimen.tif")
OUT_TIF_2 = os.path.join(BASE_DIR, "infer_jimen2.tif")


# ============ 工具函数 ============
def _all_invalid(img_patch: np.ndarray, nodata: float | int | None) -> bool:
    """判断图像 patch 是否为无效区域（全为 nodata 或 NaN）。
    img_patch: (C, H, W)
    """
    if nodata is None:
        mask_invalid = (img_patch == 0) | np.isnan(img_patch)
    else:
        mask_invalid = (img_patch == nodata) | np.isnan(img_patch)
    return bool(np.all(mask_invalid))


def _save_patch(dst_path: str, src_profile: dict, window: Window, patch: np.ndarray, count: int | None = None):
    """按照 window 写出 patch，保留地理参考。patch 形状可为 (C, H, W) 或 (H, W)。"""
    profile = src_profile.copy()
    # 统一成 (count, H, W)
    if patch.ndim == 2:
        patch_to_write = patch[np.newaxis, :, :]
        out_count = 1 if count is None else count
    else:
        patch_to_write = patch
        out_count = patch.shape[0] if count is None else count

    profile.update({
        'height': patch_to_write.shape[1],
        'width': patch_to_write.shape[2],
        'count': out_count,
        'transform': rasterio.windows.transform(window, src_profile['transform'])
    })
    # 避免出现压缩或 tiled 不兼容导致的写入问题
    for k in ['tiled', 'blockxsize', 'blockysize', 'compress', 'photometric']:
        if k in profile:
            profile.pop(k)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with rasterio.open(dst_path, 'w', **profile) as dst:
        dst.write(patch_to_write)


# ============ 1) 裁剪 ============
def crop_image_and_region(
    image_path: str,
    region_path: str,
    out_image_dir: str,
    out_region_dir: str,
    meta_json_path: str,
    tile: int = 512,
) -> dict:
    """对输入影像与区域引导图进行 512×512 裁剪。
    规则：若图像 patch 全为 0/NaN（或 nodata/NaN），则跳过；
          对于保留下来的图像 patch，若对应区域引导图为 NoData，则替换为 1。
    输出：
      - out_image_dir / out_region_dir 写出 y{row}_x{col}.tif
      - meta_json_path 保存原图尺寸、CRS、transform、有效 window 列表等信息
    """
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_region_dir, exist_ok=True)
    os.makedirs(os.path.dirname(meta_json_path), exist_ok=True)

    positions: List[Dict] = []

    with rasterio.open(image_path) as src_img, rasterio.open(region_path) as src_reg:
        if (src_img.width != src_reg.width) or (src_img.height != src_reg.height):
            raise ValueError("输入影像与区域引导图的尺寸不一致！")

        width, height = src_img.width, src_img.height
        transform = src_img.transform
        crs = src_img.crs
        img_profile = src_img.profile
        reg_profile = src_reg.profile

        img_nodata = src_img.nodata
        reg_nodata = src_reg.nodata

        # 仅处理完整 tile；边缘不足 tile 的区域忽略（保持为 0）
        for top in range(0, height - tile + 1, tile):
            for left in range(0, width - tile + 1, tile):
                window = Window(left, top, tile, tile)
                # 读取所有波段，保持原始通道数
                img_patch = src_img.read(window=window)  # (C, H, W)
                reg_patch = src_reg.read(1, window=window)  # (H, W)

                # 跳过全无效图像 patch
                if _all_invalid(img_patch, img_nodata):
                    continue

                # 区域引导图：图像有效时，将“不在 Regions 或 NoData/NaN”的值替换为 1
                valid_vals = np.array(Regions)
                mask_invalid_reg = ~np.isin(reg_patch, valid_vals)
                if reg_nodata is not None:
                    mask_invalid_reg = mask_invalid_reg | (reg_patch == reg_nodata)
                mask_invalid_reg = mask_invalid_reg | np.isnan(reg_patch)
                if np.any(mask_invalid_reg):
                    reg_patch = reg_patch.copy()
                    reg_patch[mask_invalid_reg] = 1

                # 写出裁剪块
                name = f"y{top}_x{left}.tif"
                _save_patch(os.path.join(out_image_dir, name), img_profile, window, img_patch)
                _save_patch(os.path.join(out_region_dir, name), reg_profile, window, reg_patch, count=1)

                positions.append({"y": int(top), "x": int(left), "h": tile, "w": tile, "name": name})

        meta = {
            "image_path": image_path,
            "region_path": region_path,
            "width": int(width),
            "height": int(height),
            "tile": int(tile),
            "transform_gdal": list(transform.to_gdal()),
            "crs_wkt": crs.to_wkt() if crs is not None else None,
            "positions": positions,
            "img_dtype": str(img_profile.get('dtype', 'unknown')),
        }
        with open(meta_json_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"✅ 裁剪完成：有效 patch 数量 = {len(positions)}")
        return meta


# ============ 2) 推理（复用 infer_test 思路） ============
def run_inference_on_patches(
    model_path: str,
    input_dir: str,
    region_dir: str,
    output_dir: str,
    num_classes: int,
    image_suffix=(".tif", ".tiff", ".png")
):
    os.makedirs(output_dir, exist_ok=True)
    print("🚀 开始推理（label_and_save_with_model）...")
    # 直接调用用户已有的推理函数
    label_and_save_with_model(
        model_path,
        input_dir,
        region_dir,
        output_dir,
        num_classes,
        device=None,
        image_suffix=image_suffix
    )
    print("✅ 推理完成")


# ============ 3) 拼接 ============
def _find_pred_for_tile(pred_dir: str, tile_basename_wo_ext: str) -> str | None:
    """在 pred_dir 中查找对应 tile 的预测文件。
    兼容常见扩展名与可能额外的命名后缀（如 *_pred、*_label 等）。
    优先匹配同名同后缀，其次模糊包含关系。
    """
    # 精确匹配（同名不同后缀）
    for ext in (".tif", ".tiff", ".png", ".tif.gz"):
        p = os.path.join(pred_dir, tile_basename_wo_ext + ext)
        if os.path.isfile(p):
            return p
    # 模糊匹配
    patterns = [
        f"{tile_basename_wo_ext}_*.tif",
        f"{tile_basename_wo_ext}_*.tiff",
        f"*{tile_basename_wo_ext}*.tif",
        f"*{tile_basename_wo_ext}*.tiff",
        f"*{tile_basename_wo_ext}*.png",
    ]
    for pat in patterns:
        cand = glob.glob(os.path.join(pred_dir, pat))
        if cand:
            return cand[0]
    return None


def stitch_predictions(
    meta_json_path: str,
    pred_dir: str,
    out_tif_1: str,
    out_tif_2: str,
):
    with open(meta_json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    width = int(meta['width'])
    height = int(meta['height'])
    tile = int(meta['tile'])
    transform = Affine.from_gdal(*meta['transform_gdal'])
    crs_wkt = meta.get('crs_wkt')
    crs = CRS.from_wkt(crs_wkt) if crs_wkt else None

    # 输出为二值 uint8
    canvas1 = np.zeros((height, width), dtype=np.uint8)
    canvas2 = np.zeros((height, width), dtype=np.uint8)

    # 遍历有效的 tile
    miss_count = 0
    for pos in meta['positions']:
        name = pos['name']  # y{top}_x{left}.tif
        base_wo_ext = os.path.splitext(name)[0]
        pred_path = _find_pred_for_tile(pred_dir, base_wo_ext)
        if pred_path is None:
            miss_count += 1
            continue
        with rasterio.open(pred_path) as src_pred:
            # 读取首个波段作为类别/标签
            pred = src_pred.read(1)
        # 二值化：仅保留值==1
        bin_pred = (pred == 1).astype(np.uint8)
        top = int(pos['y']); left = int(pos['x'])
        canvas1[top:top+tile, left:left+tile] = pred
        canvas2[top:top+tile, left:left+tile] = bin_pred

    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'uint8',
        'transform': transform,
    }
    if crs is not None:
        profile['crs'] = crs

    os.makedirs(os.path.dirname(out_tif_1), exist_ok=True)
    with rasterio.open(out_tif_1, 'w', **profile) as dst:
        dst.write(canvas1, 1)
    with rasterio.open(out_tif_2, 'w', **profile) as dst:
        dst.write(canvas2, 1)

    print(f"✅ 拼接完成：{out_tif_1}，{out_tif_2}；缺失预测块 {miss_count} 个")


# ============ 主流程 ============
def main():
    # 1) 裁剪
    crop_image_and_region(
        IMAGE_PATH,
        REGION_PATH,
        INFER_IMAGE_DIR,
        INFER_REGION_DIR,
        META_JSON,
        tile=PATCH_SIZE,
    )

    # 2) 推理（对裁剪后的 patch）
    run_inference_on_patches(
        MODEL_PATH,
        INFER_IMAGE_DIR,
        INFER_REGION_DIR,
        PRED_DIR,
        num_classes,
        image_suffix=(".tif",),
    )

    # 3) 拼接
    stitch_predictions(
        META_JSON,
        PRED_DIR,
        OUT_TIF_1,
        OUT_TIF_2,
    )


if __name__ == "__main__":
    main()
