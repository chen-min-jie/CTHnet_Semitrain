
# SemiTrain3_test_aligned.py
# Align student training and pseudo-label inference with train_test.py style.
import os, math, json, shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import rasterio
from rasterio.transform import from_origin

from config import (
    num_classes,
    batch_size,
    learning_rate,
    # for teacher alignment & loss
    lambda_soft,
    initial_region_prior,
)
from infer_test import label_and_save_with_model

from train_test import train_teacher, check_nan_hook, WarmupCosineScheduler  # reuse same hooks/scheduler
from utils import *
from CTHnet import CTHNet
from dataset import CroplandDataset_1, CroplandDatasetUnlabeled, HybridDataset


# ----------------------------- IO helpers -----------------------------
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


# ----------------------------- Pseudo labeling (CT) -----------------------------
@torch.no_grad()
def pseudo_label_with_ct(checkpoint_CT, device, unlabeled_imgs, checkpoint_dir, top_fraction=0.5):
    """
    Compute consensus-based confidence (CT) across three teacher checkpoints (ckpt_1/2/3),
    keep Top-K fraction, and save pseudo labels as GeoTIFF. Robust to unlabeled dataset
    not returning region_id/region_map by falling back to default region_id=0.
    """
    # 1) Unlabeled loader
    un_ds = CroplandDatasetUnlabeled(unlabeled_imgs)  # expected to return image[, region_map, region_id]
    unlisted = sorted([f for f in os.listdir(unlabeled_imgs) if f.lower().endswith(('.tif', '.tiff', '.png'))])
    un_ld = DataLoader(un_ds, batch_size=1, shuffle=False, num_workers=2)

    # 2) Load 3 teacher checkpoints
    ckpt_paths = sorted([f for f in os.listdir(checkpoint_CT) if f.endswith('.pth')])
    # Strong expectation they include: ckpt_1.pth, ckpt_2.pth, ckpt_3.pth
    models = {}
    for pth in ckpt_paths:
        m = CTHNet(num_classes=num_classes, initial_region_prior=initial_region_prior)
        bn_to_gn(m, prefer_groups=32)
        state = torch.load(os.path.join(checkpoint_CT, pth), map_location=device)
        m.load_state_dict(state if isinstance(state, dict) else state['model_state_dict'])
        m.to(device).eval()
        models[pth] = m

    all_ct = []
    plabels3 = []
    keep_ids = []

    for idx, batch in enumerate(tqdm(un_ld, desc="Calc CT")):
        # Unpack flexible outputs
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                img, region_map, region_id = batch
            elif len(batch) == 2:
                img, region_map = batch
                b = img.shape[0]
                region_id = torch.zeros(b, dtype=torch.long)
            else:
                img = batch[0]
                b = img.shape[0]
                region_map = None
                region_id = torch.zeros(b, dtype=torch.long)
        else:
            img = batch
            b = img.shape[0]
            region_map = None
            region_id = torch.zeros(b, dtype=torch.long)

        img = img.to(device)
        region_id = region_id.to(device)

        # Forward through three teachers
        logits = {}
        for t, m in models.items():
            # CTHNet forward expects (x, region_id)
            out = m(img, region_id)
            logits[t] = out["main"].softmax(1).cpu()

        # Argmax predictions
        pred = {t: l.argmax(1).squeeze(0) for t, l in logits.items()}
        # Compute CT = mIoU(M1,M3) + mIoU(M2,M3)
        # Use file names to access keys. If names don't match, fallback to first/last.
        keys = sorted(list(pred.keys()))
        key1 = 'ckpt_1.pth' if 'ckpt_1.pth' in pred else keys[0]
        key2 = 'ckpt_2.pth' if 'ckpt_2.pth' in pred else keys[1] if len(keys) > 2 else keys[0]
        key3 = 'ckpt_3.pth' if 'ckpt_3.pth' in pred else keys[-1]

        ct = miou(pred[key1], pred[key3], num_classes) + miou(pred[key2], pred[key3], num_classes)
        all_ct.append(ct)
        plabels3.append(pred[key3].numpy())
        keep_ids.append(idx)

    # 3) Keep top fraction and split high/low confidence
    all_ct = torch.tensor(all_ct)
    n_keep = max(1, int(len(all_ct) * top_fraction))
    topk = torch.topk(all_ct, k=n_keep).indices.tolist()
    lowk = torch.topk(all_ct, k=len(all_ct) - n_keep, largest=False).indices.tolist()

    # Create dirs
    high_dir = os.path.join(checkpoint_dir, "pseudo_labels_high")
    high_unlabel = os.path.join(checkpoint_dir, "pseudo_Unlabels_high")
    low_dir = os.path.join(checkpoint_dir, "pseudo_labels_low")
    low_unlabel = os.path.join(checkpoint_dir, "pseudo_Unlabels_low")
    os.makedirs(high_dir, exist_ok=True)
    os.makedirs(low_dir, exist_ok=True)
    os.makedirs(high_unlabel, exist_ok=True)
    os.makedirs(low_unlabel, exist_ok=True)

    high_meta, low_meta = [], []

    # Save high-confidence labels as GeoTIFF
    for local_idx in topk:
        global_idx = keep_ids[local_idx]
        base_name = os.path.splitext(unlisted[global_idx])[0]
        tif_path = os.path.join(high_dir, f"{base_name}.tif")
        ref_path = os.path.join(unlabeled_imgs, unlisted[global_idx])
        save_mask_tif(plabels3[local_idx], ref_path, tif_path)

        high_meta.append({"img_idx": global_idx, "pseudo": tif_path, "ct": float(all_ct[local_idx])})
        shutil.copy(os.path.join(unlabeled_imgs, unlisted[global_idx]), high_unlabel)

    # Prepare low-confidence pool (image copies + meta.json, labels will be filled later)
    for local_idx in lowk:
        global_idx = keep_ids[local_idx]
        base_name = os.path.splitext(unlisted[global_idx])[0]
        tif_path = os.path.join(low_dir, f"{base_name}.tif")  # to be generated later
        low_meta.append({"img_idx": global_idx, "pseudo": tif_path, "ct": float(all_ct[local_idx])})
        shutil.copy(os.path.join(unlabeled_imgs, unlisted[global_idx]), low_unlabel)

    with open(os.path.join(high_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(high_meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(low_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(low_meta, f, indent=2, ensure_ascii=False)

    print(f"CT Top{int(top_fraction*100)}% high: {len(high_meta)}/{len(un_ds)} → {high_dir}")
    print(f"CT Bottom{100-int(top_fraction*100)}% low: {len(low_meta)}/{len(un_ds)} → {low_dir}")

    return high_dir, low_dir, high_unlabel, low_unlabel


# ----------------------------- STA student training -----------------------------
def train_student_with_sta(
    meta_entries,  # list of dicts from meta.json (paths to pseudo labels)
    epochs,
    device,
    r,  # round index
    supervised_img,
    supervised_lbl,
    region_map_dir,
    unlabeled_imgs,
    checkpoint_Student,
):
    """
    Train the student model with STA on (supervised + high-confidence pseudo-labeled) data.
    Aligned with train_test.py: Adam + WarmupCosineScheduler, CE(ignore=255), NaN hook, BN->GN, grad clipping.
    Uses region_map + region_id if provided by dataset; otherwise falls back to zero-region soft guidance.
    """
    # ---------- augmentations for STA -----------
    simclr_aug = get_simclr_augmentations(image_size=256, use_blur=True, use_cutout=True)

    # HybridDataset should ideally return: (img, lbl, region_map, region_id). We'll handle flexible outputs.
    sta_ds = HybridDataset(supervised_img, supervised_lbl,region_map_dir,meta_entries, unlabeled_imgs, transform=simclr_aug)
    sta_ld = DataLoader(sta_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = CTHNet(num_classes=num_classes, initial_region_prior=initial_region_prior)
    bn_to_gn(model, prefer_groups=32)
    model.to(device)

    criterion_cls = torch.nn.CrossEntropyLoss(ignore_index=255)

    total_iters = epochs * max(1, len(sta_ld))
    warmup_iters = int(0.1 * total_iters)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = WarmupCosineScheduler(optimizer, warmup_iters=warmup_iters, total_iters=total_iters)

    # Register NaN hooks
    for _, module in model.named_modules():
        module.register_forward_hook(check_nan_hook)

    lrs = []
    os.makedirs(checkpoint_Student, exist_ok=True)

    for ep in range(epochs):
        model.train()
        loop = tqdm(sta_ld, desc=f"[STA r{r}] Epoch [{ep + 1}/{epochs}]")
        running_loss = 0.0

        for batch in loop:
            # Flexible unpacking: (imgs, lbls[, region_map, region_id])
            if isinstance(batch, (list, tuple)):
                if len(batch) == 4:
                    imgs, lbls, region_map, region_id = batch
                elif len(batch) == 3:
                    imgs, lbls, region_map = batch
                    b = imgs.shape[0]
                    region_id = get_region_id_from_map(lbls)
                else:
                    imgs, lbls = batch
                    b, h, w = lbls.shape
                    region_map = torch.zeros(b, h, w, dtype=torch.long)
                    region_id = get_region_id_from_map(lbls)
            else:
                raise ValueError("HybridDataset should return at least (image, label).")

            imgs = imgs.to(device)
            lbls = lbls.to(device)
            region_map = region_map.to(device) if region_map is not None else torch.zeros_like(lbls, device=device)
            region_id = region_id.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs, region_id)  # forward per train_test/CTHNet

            # Compute aligned loss: CE + lambda_soft * soft_region_guidance
            loss = model.compute_loss(outputs, lbls, lambda_soft, region_map, criterion_cls=criterion_cls)

            if torch.isnan(loss):
                print("❌ Loss became NaN")
                print("Images stats: ", imgs.min().item(), imgs.max().item(), imgs.mean().item())
                print("Labels unique: ", torch.unique(lbls))
                continue

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            lrs.append(optimizer.param_groups[0]['lr'])

        torch.cuda.empty_cache()
        avg_loss = running_loss / max(1, len(sta_ld))
        print(f"✅ Epoch {ep + 1} Done, Avg Loss: {avg_loss:.4f}")

        # Save epoch checkpoint (student)
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, os.path.join(checkpoint_Student, f"student_r{r}_epoch_{ep + 1}.pth"))

    # Save LR schedule
    with open(os.path.join(checkpoint_Student, f"student_r{r}_lr_schedule.csv"), "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["step", "learning_rate"])
        for i, lr in enumerate(lrs):
            writer.writerow([i, lr])

    # Final save
    final_path = os.path.join(checkpoint_Student, f"student_{r}.pth")
    torch.save(model.state_dict(), final_path)
    print(f"  ↳ saved {final_path}")
    return final_path


# ----------------------------- main flow -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- You can override these paths before running main() ----
    supervised_img = r"E:\test\20250727\train_image"
    supervised_lbl = r"E:\test\20250727\train_label"
    region_map_dir = r"E:\test\20250731\train_image"
    unlabeled_imgs = r"E:/test/影像/extract_cm/20250726/420081/output_unlabeled_dir"
    local_batch_size = batch_size
    epochs_supervised = 60
    local_lr = learning_rate
    local_num_classes = num_classes
    checkpoint_dir = r"E:/test/Model_cm"
    checkpoint_CT = r"E:/test/model_cm/Teacher_train/checkpoint_CT"
    checkpoint_Student = r"E:/test/model_cm/checkpoint_Student"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_Student, exist_ok=True)

    # 1) (Optional) Train teacher exactly like train_test.py if needed
    # train_teacher(supervised_img=supervised_img, supervised_lbl=supervised_lbl,
    #               num_classes=local_num_classes,
    #               checkpoint_dir=checkpoint_dir, checkpoint_CT=checkpoint_CT)

    # 2) Compute pseudo labels & split by confidence (or reuse precomputed dirs)
    # high_dir, low_dir, high_unlabel, low_unlabel = pseudo_label_with_ct(checkpoint_CT, device, unlabeled_imgs, checkpoint_dir, top_fraction=0.5)
    high_dir, low_dir, high_unlabel, low_unlabel = (
        r"E:/test/model_cm/pseudo_labels_high",
        r"E:/test/model_cm/pseudo_labels_low",
        r"E:/test/model_cm/pseudo_Unlabels_high",
        r"E:/test/model_cm/pseudo_Unlabels_low",
    )

    with open(os.path.join(high_dir, "meta.json"), "r", encoding="utf-8") as f:
        stable_high = json.load(f)
    with open(os.path.join(low_dir, "meta.json"), "r", encoding="utf-8") as f:
        stable_low = json.load(f)

    # 3) STA round-1 with high-confidence labels
    s1_path = train_student_with_sta(stable_high, 20, device, 1, supervised_img, supervised_lbl,region_map_dir,unlabeled_imgs, checkpoint_Student)

    # 4) Use student-1 to label low-confidence pool
    label_and_save_with_model(
        model_path=s1_path,
        input_dir=low_unlabel,
        output_dir=low_dir,
        model_class=CTHNet,
        num_classes=local_num_classes,
        device=device,
        image_suffix=(".tif", ".tiff", ".png"),
        initial_region_prior=initial_region_prior,  # ensure model init matches
    )

    # 5) STA round-2 with (high + newly labeled low)
    all_meta = stable_high + stable_low
    train_student_with_sta(all_meta, 20, device, 2, supervised_img, supervised_lbl,region_map_dir,unlabeled_imgs, checkpoint_Student)


if __name__ == "__main__":
    main()
