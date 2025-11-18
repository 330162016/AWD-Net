# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import cv2
import numpy as np
import torch
from torch import nn  # 保持与你原文件一致的导入风格
from utils import *   # 依赖 to_var()
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam


def _pick_rgb_indices(n_bands: int):

    table = {
        145: (47, 14, 3),   # Botswana
        103: (66, 28, 0),   # PaviaU
        102: (66, 28, 0),   # Pavia
        176: (28, 14, 3),   # KSC
        162: (25, 10, 0),   # Urban
        191: (54, 34, 10),  # Washington
        200: (28, 14, 3),   # IndianP
         64: (28, 14, 3),   # MUUFL_HSI
        144: (28, 14, 3),   # Houston_HSI
        204: (28, 14, 3),   # Salinas_corrected
    }
    return table.get(n_bands, (28, 14, 3))


def _cube_to_rgb_uint8(cube_chw: np.ndarray, r: int, g: int, b: int) -> np.ndarray:
    """

    """
    bch = cube_chw[b, :, :][:, :, np.newaxis]
    gch = cube_chw[g, :, :][:, :, np.newaxis]
    rch = cube_chw[r, :, :][:, :, np.newaxis]
    img = np.concatenate((bch, gch, rch), axis=2).astype(np.float32)
    vmin, vmax = float(img.min()), float(img.max())
    if vmax > vmin:
        img = 255.0 * (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)
    return img.clip(0, 255).astype(np.uint8)


def validate(test_list, arch, model, epoch, n_epochs):
    """

    """
    test_ref, test_lr, test_hr = test_list
    model.eval()

    psnr = 0
    with torch.no_grad():
        # Set mini-batch dataset
        ref = to_var(test_ref).detach()
        lr  = to_var(test_lr).detach()
        hr  = to_var(test_hr).detach()

        # 前向
        if arch == 'SSRNet':
            out, _, _, _, _, _ = model(lr, hr)
        elif arch == 'SSRSpat':
            _, out, _, _, _, _ = model(lr, hr)
        elif arch == 'SSRSpec':
            _, _, out, _, _, _ = model(lr, hr)
        else:
            out, _, _, _, _, _ = model(lr, hr)


        ref_np = ref.detach().cpu().numpy()
        out_np = out.detach().cpu().numpy()
        rmse  = calc_rmse(ref_np, out_np)
        psnr  = calc_psnr(ref_np, out_np)
        ergas = calc_ergas(ref_np, out_np)
        sam   = calc_sam(ref_np, out_np)

        # 追加写日志
        with open('ConSSFCNN.txt', 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{rmse},{psnr},{ergas},{sam},\n")

        # ===== 可视化与保存 =====
        try:
            os.makedirs('./figs', exist_ok=True)

            # 选择可视化波段索引
            n_bands = int(ref_np.shape[1])  # [B,C,H,W] -> C
            r_idx, g_idx, b_idx = _pick_rgb_indices(n_bands)

            # 取第一个样本
            ref_chw = np.squeeze(ref_np[0])  # [C,H,W]
            out_chw = np.squeeze(out_np[0])  # [C,H,W]

            # 转伪 RGB（各自独立归一化）
            ref_rgb = _cube_to_rgb_uint8(ref_chw, r_idx, g_idx, b_idx)
            out_rgb = _cube_to_rgb_uint8(out_chw, r_idx, g_idx, b_idx)

            # 尺寸对齐
            H, W = out_rgb.shape[:2]
            if ref_rgb.shape[:2] != (H, W):
                ref_rgb = cv2.resize(ref_rgb, (W, H), interpolation=cv2.INTER_NEAREST)

            # 保存 PNG
            pre_path  = f'./figs/epoch{epoch:04d}_pre_HR-HSI.png'
            post_path = f'./figs/epoch{epoch:04d}_post_SR-HSI_{arch}.png'
            cv2.imwrite(pre_path,  ref_rgb)
            cv2.imwrite(post_path, out_rgb)

            # 保存融合后 SR-HSI 立方体为 .mat
            try:
                from scipy.io import savemat
                mat_path = f'./figs/epoch{epoch:04d}_post_SR-HSI_{arch}.mat'
                # 以 CHW 保存，避免引入额外轴；MATLAB 读取后为 CxHxW
                savemat(mat_path, {'SR_HSI': out_chw})
            except Exception as e:
                print(f"[warn] 保存.mat失败：{e}")

        except Exception as e:
            print(f"[warn] 保存可视化失败：{e}")

    return psnr
