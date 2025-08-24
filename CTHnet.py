from __future__ import annotations

import json
import os

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from tqdm import tqdm

from config import num_classes, num_region_types
from utils import bn_to_gn, build_soft_region_label, soft_cross_entropy

try:
    import timm  # backbone & Swin 实现
except ImportError as e:
    raise ImportError("Please install timm first: pip install timm") from e

from timm.models.swin_transformer import SwinTransformerBlock, PatchMerging

class SemanticExtractor(nn.Module):
    """
    480‑ch 输入 → BasicLayer(depth=2) × 3 + 最后 2 block
    共 8 Swin blocks，产生 T1..T4 (与 C1..C4 对齐)
    """
    def __init__(self, window_size=4, mlp_ratio=4.,H0=8,W0=8):
        super().__init__()
        # Stage‑0 (same res as CT4) — 2 blocks
        self.stage0 = nn.Sequential(
            SwinTransformerBlock(dim=480, num_heads=8, window_size=window_size,
                                 shift_size=0, mlp_ratio=mlp_ratio,input_resolution=(H0, W0)),
            SwinTransformerBlock(dim=480, num_heads=8, window_size=window_size,
                                 shift_size=window_size//2, mlp_ratio=mlp_ratio,input_resolution=(H0, W0)),
        )
        # Patch‑Merge ↓2  → 960 ch
        # self.merge1 = PatchMerging(480, norm_layer=nn.LayerNorm)
        # Stage‑1
        self.stage1 = nn.Sequential(
            SwinTransformerBlock(dim=480, num_heads=16, window_size=window_size,
                                 shift_size=0, mlp_ratio=mlp_ratio,input_resolution=(H0, W0)),
            SwinTransformerBlock(dim=480, num_heads=16, window_size=window_size,
                                 shift_size=window_size//2, mlp_ratio=mlp_ratio,input_resolution=(H0, W0)),
        )
        # self.merge2 = PatchMerging(960, norm_layer=nn.LayerNorm)     # ↓2 → 1920
        # Stage‑2
        self.stage2 = nn.Sequential(
            SwinTransformerBlock(dim=480, num_heads=32, window_size=window_size,
                                 shift_size=0, mlp_ratio=mlp_ratio,input_resolution=(H0, W0)),
            SwinTransformerBlock(dim=480, num_heads=32, window_size=window_size,
                                 shift_size=window_size//2, mlp_ratio=mlp_ratio,input_resolution=(H0, W0)),
        )
        # self.merge3 = PatchMerging(1920, norm_layer=nn.LayerNorm)    # ↓2 → 3840
        # Stage‑3 (最后 2 block，无再降采)
        self.stage3 = nn.Sequential(
            SwinTransformerBlock(dim=480, num_heads=32, window_size=window_size,
                                 shift_size=0, mlp_ratio=mlp_ratio,input_resolution=(H0, W0)),
            SwinTransformerBlock(dim=480, num_heads=32, window_size=window_size,
                                 shift_size=window_size//2, mlp_ratio=mlp_ratio,input_resolution=(H0, W0)),
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()   # → (B,H0,W0,480)
        t1 = self.stage0(x)                   # same res as CT4
        # x  = self.merge1(t1)                  # 1/2 res
        t2 = self.stage1(t1)
        # x  = self.merge2(t2)                  # 1/4 res
        t3 = self.stage2(t2)
        # x  = self.merge3(t3)                  # 1/8 res
        t4 = self.stage3(t3)
        # ★ 把每级 NHWC → NCHW，供 AFM 进行 Conv2d
        t1 = t1.permute(0, 3, 1, 2).contiguous()
        t2 = t2.permute(0, 3, 1, 2).contiguous()
        t3 = t3.permute(0, 3, 1, 2).contiguous()
        t4 = t4.permute(0, 3, 1, 2).contiguous()
        return [t1, t2, t3, t4]               # 对应 T1..T4

# --------------------------- Helper blocks --------------------------- #
class ConvBNAct(nn.Sequential):
    """1×1 or 3×3 Conv + BN + ReLU"""
    def __init__(self, in_c: int, out_c: int, k: int = 1):
        padding = 0 if k == 1 else 1
        super().__init__(
            nn.Conv2d(in_c, out_c, k, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

class AFM(nn.Module):
    """Adaptive Fusion Module ‑‑ 论文 Fig‑3 完整实现 (含残差)"""

    def __init__(self, c_cnn: int, c_tr: int, out_c: int):
        super().__init__()
        # ① 通道对齐
        self.conv_c = nn.Conv2d(c_cnn, out_c, 1, padding=0)
        self.conv_t = nn.Conv2d(c_tr,  out_c, 1,padding=0)
        # ② 合并后特征 → 2C
        self.merge_conv = nn.Conv2d(out_c * 2, out_c * 2, 1,padding=0)
        self.merge_conv1 = nn.Conv2d(c_cnn + c_tr, out_c , 1,padding=0)#使用残差连接
        # ③ 各流权重生成
        self.weight_c = nn.Sequential(nn.Conv2d(out_c, out_c, 1, bias=False), nn.Sigmoid())
        self.weight_t = nn.Sequential(nn.Conv2d(out_c, out_c, 1, bias=False), nn.Sigmoid())

    def forward(self, feat_c: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
        # Step‑1 对齐通道
        fc = self.conv_c(feat_c)
        ft = self.conv_t(feat_t)

        # Step‑2 合并→2C，再 1×1 处理
        merged = torch.cat([fc, ft], dim=1)
        merged = self.merge_conv(merged)
        mc, mt = torch.chunk(merged, 2, dim=1)  # Split 2 路

        # Step‑3 Sigmoid 权重
        wc = self.weight_c(mc)
        wt = self.weight_t(mt)

        # Step‑4 Softmax 归一化 (跨分支维度)
        w = torch.stack([wc, wt], dim=1)          # (B,2,C,H,W)
        w = torch.softmax(w, dim=1)
        wc, wt = w[:, 0], w[:, 1]

        # step-5 使用残差连接(虚线)添加先前的特征
        merged1 = torch.cat([feat_c, feat_t], dim=1)
        merged1 = self.merge_conv1(merged1)

        # Step‑5 加权融合 + 残差 (fc, ft)
        fused = fc * wc + ft * wt
        fused = fused + merged1                # ← 论文中虚线残差连接
        return fused


# --------------------------- Segmentation Head --------------------------- #
class SegHead(nn.Module):
    """论文图示分割头：CT1‑4 上采样对齐 → concat → 1×1 Conv + BN + ReLU6 → 1×1 Conv"""
    def __init__(self, in_channels: list[int], mid_c: int, num_classes: int):
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(sum(in_channels), mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU6(inplace=True),
        )
        self.classifier = nn.Conv2d(mid_c, num_classes, 1)

    def forward(self, feats: List[torch.Tensor]):
        # 取 CT1 的分辨率为基准
        base_h, base_w = feats[0].shape[-2:]
        ups = [F.interpolate(f, size=(base_h, base_w), mode="bilinear", align_corners=False)
               if f.shape[-2:] != (base_h, base_w) else f for f in feats]
        x = torch.cat(ups, dim=1)
        x = self.pre_conv(x)
        return self.classifier(x)



# ----------------------------- CTHNet ----------------------------- #
class CTHNet(nn.Module):
    def __init__(self, num_classes: int = num_classes,
                 cnn_backbone: str = "res2net50_26w_4s",
                 pretrained: bool = True,
                 lambda_bd: float = 0.1, initial_region_prior=None):
        super().__init__()
        self.lambda_bd = lambda_bd
        # 如果 initial_region_prior 是 None，给它一个默认值
        if initial_region_prior is None:
            # 这里可以根据实际情况设定默认值
            initial_region_prior = np.ones((num_region_types, num_classes)) * 0.01  # 默认值

        # ✅ 第一步：初始化时添加 region embedding 模块
        self.region_embed = nn.Embedding(num_region_types, embedding_dim=480)  # 你Transformer使用的是480维
        self.use_attention_region = True  # 控制是否使用 attention 融合

        # ✅ 新增：初始化 learnable region prior 参数表（每个区域对应一个类别分布）
        self.region_prior_param = nn.Parameter(torch.tensor(initial_region_prior))
        # initial_region_prior: shape = (num_region_types, num_classes)，由人工预设提供

        
        # 1) CNN encoder
        self.cnn = timm.create_model(cnn_backbone, features_only=True, pretrained=pretrained,
                                     out_indices=(0,1, 2, 3))  # C1‑C4
        c_channels = self.cnn.feature_info.channels()

        # (0) 预合并 1×1 Conv 把 CT1..4 → 480ch
        self.pre_merge = nn.Sequential(
            nn.Conv2d(sum(c_channels), 480, 1, bias=False),
            nn.BatchNorm2d(480),
            nn.ReLU(inplace=True)
        )

        # (1) 替换 timm swin 为自定义 SemanticExtractor
        self.transformer = SemanticExtractor(H0=32, W0=32)
        t_channels = [480, 480, 480, 480]



        # 3) AFM 融合
        self.afms = nn.ModuleList([
            AFM(c_c, t_c, out_c=c_c) for c_c, t_c in zip(c_channels, t_channels)
        ])

        # 4) Segmentation head
        self.seg_head = SegHead(c_channels, mid_c=64, num_classes=num_classes)

        # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, region_id: torch.Tensor):
        # CNN features
        c_feats = self.cnn(x)  # list [C1 .. C4]

        # Transformer features (同样有 4 个尺度)
        # ---- 把全部特征对齐到 C4 分辨率 (1/32) 并 concat ----
        base_size = c_feats[-1].shape[-2:]  # C4 尺寸
        ups = [F.interpolate(f, size=base_size, mode="bilinear", align_corners=False)
               if f.shape[-2:] != base_size else f
               for f in c_feats]  # list 长度=4
        merged = torch.cat(ups, dim=1)  # (B, ΣC, H/32, W/32)
        # ---- 压缩到 480 通道，作为 Swin 输入 ----
        # ✅ 第三步：获取 region_token，并加到 Transformer 输入上
        B = region_id.shape[0]
        region_id =region_id.long()
        region_token = self.region_embed(region_id).view(B, 480, 1, 1)  # (B,480,1,1)
        x_t = self.pre_merge(merged)
        if self.use_attention_region:
            # ✅ 使用 region_token 加入 Swin Transformer Attention（通道层融合）
            x_t = x_t + region_token # 再次强调区域 token 已在空间层参与注意力

        t_feats = self.transformer(x_t)  # T1..T4
        aligned_t = [F.interpolate(t, size=c.shape[-2:], mode='bilinear',
                                   align_corners=False) if t.shape[-2:] != c.shape[-2:]
                     else t
                     for t, c in zip(t_feats, c_feats)]


        # AFM 融合
        fused = [afm(c, t) for afm, c, t in zip(self.afms, c_feats, aligned_t)]  # CT1‑CT4
        c1, c2, c3, c4 = fused  # renaming

        # 主分支：分割头 (输出分辨率 = 输入 /4)，再上采到原分辨率
        H, W = x.shape[-2:]
        logits_small = self.seg_head([c1, c2, c3, c4])  # (B,num_cls,H/4,W/4)
        logits = F.interpolate(logits_small, size=(H, W), mode="bilinear", align_corners=False)



        return {
            "main": logits,
        }

# -------------- convenience loss wrapper (optional) -------------- #
    def compute_loss(
            self,
            preds: dict[str, torch.Tensor],
            target,
            lambda_soft,
            region_map: torch.Tensor,
            criterion_cls
    ):
        """
        preds: 模型预测字典，包含 "main": (B,C,H,W)
        region_map: 区域图 (B,H,W)，每个像素是一个 int 区域编号
        region_prior: 每个区域的类别概率分布 dict[int -> list[float]]
        num_classes: 类别总数
        criterion_cls: 可选的 F.cross_entropy 备用
        """
        # 修改为双重引导：
        loss_main = criterion_cls(preds["main"], target)  # 硬标签
        soft_target = build_soft_region_label(region_map, self.region_prior_param)
        loss_soft = soft_cross_entropy(preds["main"], soft_target)
        loss = loss_main + lambda_soft * loss_soft  # 总损失 = CE + soft 引导
        # 检查软标签损失是否为负
        if loss_soft < 0:
            print(f"[WARN] loss_soft < 0: {loss_soft.item()}")
            # 检查硬标签损失是否为负
        if loss_main < 0:
            print(f"[WARN] loss_main < 0: {loss_main.item()}")
        return loss



