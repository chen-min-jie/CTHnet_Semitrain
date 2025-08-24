# train_teacher_and_sta.py
import os, math, json, shutil
import rasterio
from rasterio.transform import from_origin
from sympy import false

from config import num_classes, batch_size, learning_rate
from train_test import train_teacher, check_nan_hook, WarmupCosineScheduler
from utils import *
import numpy as np
import torch, torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from CTHnet import CTHNet
from dataset import CroplandDataset_1, CroplandDatasetUnlabeled,HybridDataset


def save_mask_tif(array, reference_tif_path, save_path, dtype='uint8'):
    with rasterio.open(reference_tif_path) as src:
        profile = src.profile.copy()
        profile.update({
            'count': 1,
            'dtype': dtype,
            'nodata': 255,
            'driver': 'GTiff'
        })
        with rasterio.open(save_path, 'w', **profile) as dst:
            dst.write(array.astype(dtype), 1)

def pseudo_label_with_ct(checkpoint_CT, device, unlabeled_imgs,checkpoint_dir,top_fraction=0.5):
    # 1) 准备无标签 loader
    un_ds = CroplandDatasetUnlabeled(unlabeled_imgs)  # 只返回 image
    unlabelde=sorted([f for f in os.listdir(unlabeled_imgs) if f.endswith('.tif') or f.endswith('.png')])
    un_ld = DataLoader(un_ds, batch_size=1, shuffle=False, num_workers=2)

    # 2) 加载 3 份权重
    ckpt_paths=sorted([f for f in os.listdir(checkpoint_CT) if f.endswith('.pth')])
    models = {}
    for pth in ckpt_paths:
        m = CTHNet(num_classes=num_classes)
        bn_to_gn(m, prefer_groups=32)
        m.load_state_dict(torch.load(os.path.join(checkpoint_CT,pth), map_location=device))
        m.to(device).eval()
        models[pth] = m

    all_ct = []  # 存储每张图像的 CT 值
    plabels3 = []  # 来自最终(3/3)模型的伪标签，后续要喂给 STA
    keep_ids = []  # 对应 un_ds 中的索引

    with torch.no_grad():
        for idx, img in enumerate(tqdm(un_ld, desc="Calc CT")):
            img = img.to(device)

            # 三个模型前向
            logits = {t: models[t](img)["main"].softmax(1).cpu() for t in models}
            pred = {t: l.argmax(1).squeeze(0) for t, l in logits.items()}

            # 计算 CT：mIoU(M1, M3) + mIoU(M2, M3)
            ct = miou(pred["ckpt_1.pth"], pred["ckpt_3.pth"], num_classes) + \
                 miou(pred["ckpt_2.pth"], pred["ckpt_3.pth"], num_classes)

            all_ct.append(ct)
            plabels3.append(pred["ckpt_3.pth"].numpy())  # 保存 numpy 更省显存
            keep_ids.append(idx)

    # 3) 计算 Top 50% 样本并将其划分为高置信度（high）和低置信度（low）
    all_ct = torch.tensor(all_ct)
    n_keep = int(len(all_ct) * top_fraction)
    topk = torch.topk(all_ct, k=n_keep).indices.tolist()
    lowk = torch.topk(all_ct, k=len(all_ct) - n_keep, largest=False).indices.tolist()

    # 保存高置信度和低置信度的伪标签
    high_meta = []
    low_meta = []

    # 创建存储目录
    high_dir = os.path.join(checkpoint_dir, "pseudo_labels_high")
    high_unlabel = os.path.join(checkpoint_dir, "pseudo_Unlabels_high")
    low_dir = os.path.join(checkpoint_dir, "pseudo_labels_low")
    low_unlabel = os.path.join(checkpoint_dir, "pseudo_Unlabels_low")
    os.makedirs(high_dir, exist_ok=True)
    os.makedirs(low_dir, exist_ok=True)
    os.makedirs( high_unlabel, exist_ok=True)
    os.makedirs(low_unlabel, exist_ok=True)

    # 高置信度伪标签
    for local_idx in topk:
        global_idx = keep_ids[local_idx]
        name = un_ds.get_image_name(global_idx)
        tif_path = os.path.join(high_dir, f"{name}.tif")
        ref_path = os.path.join(unlabeled_imgs, unlabelde[global_idx])
        # ✅ 保存为 GeoTIFF
        save_mask_tif(plabels3[local_idx], ref_path, tif_path)

        high_meta.append({
            "img_idx": global_idx,
            "pseudo": tif_path,
            "ct": float(all_ct[local_idx])
        })
        shutil.copy(os.path.join(unlabeled_imgs,unlabelde[global_idx]),high_unlabel)


    # 低置信度伪标签(第二次打标签的时候会放入low_dir这个文件夹，先提前设置索引)
    for local_idx in lowk:
        global_idx = keep_ids[local_idx]
        name = un_ds.get_image_name(global_idx)
        fname = f"{name}.tif"
        np_path = os.path.join(low_dir, fname)
        # np.save(np_path, plabels3[local_idx])
        low_meta.append({"img_idx": global_idx, "pseudo": np_path, "ct": float(all_ct[local_idx])})
        shutil.copy(os.path.join(unlabeled_imgs, unlabelde[global_idx]), low_unlabel)

    # 保存高置信度和低置信度的 meta.json 文件
    with open(os.path.join(high_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(high_meta, f, indent=2)
    with open(os.path.join(low_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(low_meta, f, indent=2)

    print(f"CT Top50% high: {len(high_meta)}/{len(un_ds)} → saved to {high_dir}")
    print(f"CT Bottom50% low: {len(low_meta)}/{len(un_ds)} → saved to {low_dir}")

    # 返回高置信度和低置信度的 meta 文件路径
    return high_dir, low_dir, high_unlabel, low_unlabel
# --------------------------------------------------------------------------------------
# STA – 学生模型的训练
# --------------------------------------------------------------------------------------
def train_student_with_sta(undo_label,epochs,device,r,supervised_img,supervised_lbl,unlabeled_imgs,checkpoint_Student):
    # 读取 meta.json
    stable_meta = undo_label#json读取的meta.json文件
    # ---------- 构造强增强 ---------------
    simclr_aug = get_simclr_augmentations(image_size=256,
    use_blur=True,
    use_cutout=True)

    sta_ds = HybridDataset(supervised_img, supervised_lbl,
                           stable_meta, unlabeled_imgs,transform=simclr_aug)

    sta_ld = DataLoader(sta_ds, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    model = CTHNet(num_classes=num_classes)
    bn_to_gn(model)
    model.to(device)

    criterion_cls = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    criterion_bd  = torch.nn.BCEWithLogitsLoss()
    optimizer     = optim.Adam(model.parameters(), lr=learning_rate)

    total_iters = epochs * len(sta_ld)
    warmup_iters = int(0.1 * total_iters)  # 可调比例
    scheduler = WarmupCosineScheduler(optimizer, warmup_iters=warmup_iters, total_iters=total_iters)

    for name, module in model.named_modules():
        module.register_forward_hook(check_nan_hook)

    lrs = []
    for ep in range(epochs):        # 混合，进行训练
        model.train()
        loop = tqdm(sta_ld, desc=f"Epoch [{ep + 1}/{epochs}]")
        running = 0.
        for imgs, lbls in tqdm(sta_ld, desc=f"[STA r{r}] ep{ep}"):
            if torch.isnan(lbls).any():
                print("❌ 输入图像包含 NaN！")
                print("→ NaN 位置:", torch.nonzero(torch.isnan(lbls)))
                print("→ 图像最小值:",lbls.min().item(), "最大值:", lbls.max().item())
                raise ValueError("输入图像含 NaN，不合理，已中断训练。")

            # 替换标签中非法值（如 15）为 ignore_index=255
            lbls[~((lbls == 0) | (lbls == 1))] = 255
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out   = model(imgs)


            # edges = ( torch.abs(F.pad(lbls[:,1:] - lbls[:,:-1], (0,0,1,0))) > 0 ) | \
            #         ( torch.abs(F.pad(lbls[:,:,1:] - lbls[:,:,:-1], (1,0,0,0))) > 0 )
            # edges = edges.float().unsqueeze(1)
            loss  = model.compute_loss(out, lbls, None, criterion_cls, criterion_bd,use_boundary=False)
            if torch.isnan(loss):
                print("❌ Loss became NaN")
                print("Images stats: ", imgs.min().item(), imgs.max().item(), imgs.mean().item())
                print("Labels unique: ", torch.unique(lbls))
                continue

            loss.backward()
            optimizer.step()
            scheduler.step()  # ✅ 每步更新学习率
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        lrs.append(optimizer.param_groups[0]['lr'])
        print(f"✅ Epoch {ep + 1} Done, Avg Loss: {running / len(sta_ld):.4f}")
    torch.save(model.state_dict(), os.path.join(checkpoint_Student, f"student_{r}.pth"))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supervised_img = r"E:\test\20250727\train_image"
    supervised_lbl =  r"E:\test\20250727\train_label"
    unlabeled_imgs =  "E:/test/影像/extract_cm/20250726/420081/output_unlabeled_dir"
    batch_size = 8
    epochs_supervised = 60
    learning_rate = 1e-5
    num_classes = 2
    checkpoint_dir = "E:/test/Model_cm"
    checkpoint_CT = "E:/test/model_cm/Teacher_train/checkpoint_CT"
    checkpoint_Student = "E:/test/model_cm/checkpoint_Student"

    # train_teacher(supervised_img=supervised_img, supervised_lbl=supervised_lbl, num_classes=4,
    #               checkpoint_dir=checkpoint_dir, checkpoint_CT=checkpoint_CT)

    #2.对伪标签计算置信度
    # high_dir, low_dir, high_unlabel, low_unlabel=pseudo_label_with_ct(checkpoint_CT, device,unlabeled_imgs,checkpoint_dir,top_fraction=0.5)
    high_dir, low_dir, high_unlabel, low_unlabel = 'E:/test/model_cm/pseudo_labels_high', 'E:/test/model_cm/pseudo_labels_low',  'E:/test/model_cm/pseudo_Unlabels_high','E:/test/model_cm/pseudo_Unlabels_low'
    with open(os.path.join(high_dir, "meta.json"), "r", encoding="utf-8") as f:
        stable_high = json.load(f)#高置信度文件索引
    with open(os.path.join(low_dir, "meta.json"), "r", encoding="utf-8") as f:
        stable_low = json.load(f)  # 高置信度文件索引
    #3.使用高置信度的标签进行学生模型训练--STA
    train_student_with_sta(stable_high, 20, device, 1,supervised_img,supervised_lbl,unlabeled_imgs,checkpoint_Student)

    #4.使用第一次学生训练的模型对低置信度的样本打标签
    label_and_save_with_model(model_path = os.path.join(checkpoint_Student, "student_1.pth"),
    input_dir = low_unlabel,
    output_dir = low_dir,
    model_class = CTHNet,
    num_classes = num_classes,
    device=device,
    image_suffix=(".tif", ".tiff", ".png"))

    #5.混合模型进行Sta训练学生模型2
    all_dir = stable_high + stable_low
    train_student_with_sta(all_dir, 20, device, 2,supervised_img,supervised_lbl,unlabeled_imgs,checkpoint_Student)
if __name__ == "__main__":
    main()

