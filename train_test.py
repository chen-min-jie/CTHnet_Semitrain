import csv
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from CTHnet import CTHNet
from utils import bn_to_gn

from dataset import CroplandDataset_1
from config import supervised_img, supervised_lbl, batch_size, epochs_supervised, \
    learning_rate, num_classes, checkpoint_dir, checkpoint_CT, region_map_dir, Classes, initial_region_prior, \
    lambda_soft
from time import time
import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=1e-6, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_iters:
            return [base_lr * (step + 1) / self.warmup_iters for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_iters) / max(1, self.total_iters - self.warmup_iters)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]


# 在训练开始前一次性注册 NaN 钩子
def check_nan_hook(module, input, output):
    if isinstance(output, torch.Tensor) and torch.isnan(output).any():
        print(f"[NaN Warning] in module: {module.__class__.__name__}")
        raise RuntimeError("NaN detected in forward pass.")


#教师模型训练过程
def train_teacher(supervised_img = supervised_img,
                  supervised_lbl = supervised_lbl,
                  region_map_dir = region_map_dir,
                  batch_size = batch_size,
                  epochs_supervised = epochs_supervised,
                  learning_rate = learning_rate,
                  num_classes = num_classes,
                  checkpoint_dir = checkpoint_dir,
                  checkpoint_CT = checkpoint_CT,
                  lambda_soft = lambda_soft,
                  device = None, use_boundary=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载训练数据
    dataset = CroplandDataset_1(supervised_img, supervised_lbl, region_map_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    
    # 初始化模型、损失函数、优化器
    model = CTHNet(num_classes=num_classes,initial_region_prior=initial_region_prior)
    bn_to_gn(model, prefer_groups=32)
    model.to(device)

    # criterion = CombinedLoss()
    criterion_cls = torch.nn.CrossEntropyLoss(ignore_index=255)  # 255 忽略

    total_iters = epochs_supervised * len(data_loader)
    warmup_iters = int(0.1 * total_iters)  # 可调比例

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = WarmupCosineScheduler(optimizer, warmup_iters=warmup_iters, total_iters=total_iters)

    #用字典设置模型训练的保存节点
    milestones = {  # epoch(从1开始) : 文件名
        math.ceil(epochs_supervised / 3): "ckpt_1.pth",
        math.ceil(epochs_supervised * 2 / 3): "ckpt_2.pth",
        epochs_supervised-1: "ckpt_3.pth",
    }
    for name, module in model.named_modules():
        module.register_forward_hook(check_nan_hook)

    lrs = []
    # ============== 开始训练循环 ==============
    for epoch in range(epochs_supervised):
        model.train()
        epoch_loss = 0
        loop = tqdm(data_loader, desc=f"Epoch [{epoch + 1}/{epochs_supervised}]")

        for it, (images, labels, region_map, region_id) in enumerate(loop):  # 解包 region_map 和 region_id
            images, labels, region_map, region_id = images.to(device), labels.to(device), region_map.to(device), region_id.to(device)

            optimizer.zero_grad()


            # 前向传播
            try:
                outputs = model(images, region_id)
            except RuntimeError as e:
                print("⚠️ Forward pass error:", e)
                continue



            # 计算损失
            loss = model.compute_loss(outputs, labels, lambda_soft,region_map, criterion_cls=criterion_cls)

            if torch.isnan(loss):
                print("❌ Loss became NaN")
                print("Images stats: ", images.min().item(), images.max().item(), images.mean().item())
                print("Labels unique: ", torch.unique(labels))
                continue

            # 反向传播
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()  # ✅ 每步更新学习率

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

            lrs.append(optimizer.param_groups[0]['lr'])
        torch.cuda.empty_cache()
        print(f"✅ Epoch {epoch + 1} Done, Avg Loss: {epoch_loss / len(data_loader):.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss,
        }, os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth'))

        # torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth'))
        # 如果到达里程碑 → 保存
        if epoch in milestones:
            ckpt_path = os.path.join(checkpoint_CT, milestones[epoch])
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ saved {ckpt_path}")

        with open("learning_rate_schedule.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "learning_rate"])
            for i, lr in enumerate(lrs):
                writer.writerow([i, lr])

if __name__ == '__main__':
    output_image_dir = 'G:/20250807/origin_data/image_data'
    output_mask_dir = 'G:/20250807/origin_data/mask_data'
    output_region_map_dir = 'G:/20250807/origin_data/region_map_dir'  # 输出区域引导图路径
    output_unlabeled_dir = 'G:/20250807/origin_data/unlabeled_data'
    batch_size = batch_size
    epochs_supervised = epochs_supervised
    learning_rate = learning_rate
    num_classes = 4
    lambda_soft=lambda_soft
    checkpoint_dir = "G:/20250807/Teacher_train"
    checkpoint_CT = "G:/20250807/Teacher_train/checkpoint_CT"

    train_teacher(supervised_img=output_image_dir, supervised_lbl=output_mask_dir,region_map_dir=output_region_map_dir,num_classes=num_classes,
                  checkpoint_dir=checkpoint_dir, checkpoint_CT=checkpoint_CT, lambda_soft=lambda_soft,use_boundary=False)
    # train_teacher(supervised_img=supervised_img,supervised_lbl=supervised_lbl,num_classes = num_classes,checkpoint_dir=checkpoint_dir,checkpoint_CT=checkpoint_CT,use_boundary=False)