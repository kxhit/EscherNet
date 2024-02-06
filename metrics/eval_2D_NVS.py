import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import json
from skimage.metrics import structural_similarity as calculate_ssim
import lpips
LPIPS = lpips.LPIPS(net='alex', version='0.1')


def calc_2D_metrics(pred_np, gt_np):
    # pred_np: [H, W, 3], [0, 255], np.uint8
    pred_image = torch.from_numpy(pred_np).unsqueeze(0).permute(0, 3, 1, 2)
    gt_image = torch.from_numpy(gt_np).unsqueeze(0).permute(0, 3, 1, 2)
    # [0-255] -> [-1, 1]
    pred_image = pred_image.float() / 127.5 - 1
    gt_image = gt_image.float() / 127.5 - 1
    # for 1 image
    # pixel loss
    loss = F.mse_loss(pred_image[0], gt_image[0].cpu()).item()
    # LPIPS
    lpips = LPIPS(pred_image[0], gt_image[0].cpu()).item()  # [-1, 1] torch tensor
    # SSIM
    ssim = calculate_ssim(pred_np, gt_np, channel_axis=2)
    # PSNR
    psnr = cv2.PSNR(gt_np, pred_np)

    return loss, lpips, ssim, psnr


# script to evaluate the model on GSO/RTMV/NeRF dataset
# todo modify the path
LOG_DIR = "../logs_4DoF"      # 6DoF, 4DoF
DATASET = "NeRF" # GSO RTMV NeRF

if DATASET == "GSO":
    DATA_DIR = "/home/xin/data/EscherNet/Data/GSO30/"
    T_ins = [1, 2, 3, 5, 10]
    total_views = 25
    DATA_TYPE = "render_mvs_25"
    start_id = 10   # from 10 to total_views as test views
elif DATASET == "RTMV":
    T_ins = [1, 2, 3, 5, 10]
    total_views = 20
    start_id = 10
elif DATASET == "NeRF":
    DATA_DIR = "/home/xin/data/EscherNet/Data/nerf_synthetic/"
    T_ins = [1, 2, 3, 5, 10, 20, 50, 100]
    total_views = 200
    start_id = 0
else:
    raise NotImplementedError

for T_in in tqdm(T_ins):
    if DATASET == "GSO":
        log_dir = os.path.join(LOG_DIR, f"GSO25/N{T_in}M25")
    elif DATASET == "RTMV":
        log_dir = os.path.join(LOG_DIR, f"RTMV/N{T_in}M20")
        DATA_DIR = log_dir
    elif DATASET == "NeRF":
        log_dir = os.path.join(LOG_DIR, f"NeRF/N{T_in}M200")

    # calc 2D metrics
    val_lpips = 0
    val_ssim = 0
    val_psnr = 0
    val_loss = 0
    val_num = 0

    # get all objects
    objects = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]

    for obj in objects:
        lpips = 0
        ssim = 0
        psnr = 0
        loss = 0
        num = 0
        color = [1., 1., 1., 1.]

        if DATASET == "NeRF":
            # load test info
            with open(os.path.join(DATA_DIR, obj, "transforms_test.json"), "r") as f:
                test_info = json.load(f)["frames"]
            total_views = len(test_info)

        for i in range(start_id, total_views):
            # load the ground truth
            if DATASET == "GSO":
                gt_path = os.path.join(DATA_DIR, obj, DATA_TYPE, "model", f"{i:03d}.png")    # 001 for target view
                img = plt.imread(gt_path)
                img[img[:, :, -1] == 0.] = color
                gt = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
                gt = gt.resize((256, 256))
                gt = np.array(gt)
            elif DATASET == "RTMV":
                gt_path = os.path.join(DATA_DIR, obj, "gt.png")
                img = plt.imread(gt_path)
                gt = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
                gt = np.array(gt)
                gt = gt[:, 256 * i:256 * (i + 1), :]
            elif DATASET == "NeRF":
                img_path = os.path.join(DATA_DIR, obj, test_info[i]["file_path"] + ".png")
                img = plt.imread(img_path)
                img[img[:, :, -1] == 0.] = color
                gt = Image.fromarray(np.uint8(img[:, :, :3] * 255.)).convert("RGB")
                gt = gt.resize((256, 256))
                gt = np.array(gt)


            # load the prediction
            if DATASET == "GSO":
                pred_path = os.path.join(log_dir, obj, "0.png")
                # split width 4096 to 16 images, each 256 width
                pred = plt.imread(pred_path)
                pred = Image.fromarray(np.uint8(pred[:, :, :3] * 255.))
                # Image to numpy
                pred = np.array(pred)
                pred = pred[:, 256*i:256*(i+1), :]
            elif DATASET == "RTMV":
                pred_path = os.path.join(log_dir, obj, "0.png")
                # split width 4096 to 16 images, each 256 width
                pred = plt.imread(pred_path)
                pred = Image.fromarray(np.uint8(pred[:, :, :3] * 255.))
                pred = np.array(pred)
                pred = pred[:, 256 * i:256 * (i + 1), :]
            elif DATASET == "NeRF":
                pred_path = os.path.join(log_dir, obj, f"{i}.png")
                # read preds one by one
                pred = plt.imread(pred_path)
                pred = Image.fromarray(np.uint8(pred[:, :, :3] * 255.))
                pred = pred.resize((256, 256))
                pred = np.array(pred)

            # 2D metrics
            loss_i, lpips_i, ssim_i, psnr_i = calc_2D_metrics(pred, gt)

            loss += loss_i
            lpips += lpips_i
            ssim += ssim_i
            psnr += psnr_i
            num += 1
        loss /= num
        lpips /= num
        ssim /= num
        psnr /= num

        val_loss += loss
        val_lpips += lpips
        val_ssim += ssim
        val_psnr += psnr
        val_num += 1

        print(f"obj: {obj}, loss: {loss}, lpips: {lpips}, ssim: {ssim}, psnr: {psnr}")
        # save the results to txt
        with open(os.path.join(log_dir, "metrics2D.txt"), "a") as f:
            f.write(f"obj: {obj}, loss: {loss}, lpips: {lpips}, ssim: {ssim}, psnr: {psnr}\n")


    print(f"avg loss: {val_loss / val_num}, avg lpips: {val_lpips / val_num}, avg ssim: {val_ssim / val_num}, avg psnr: {val_psnr / val_num}")
    # save the results to txt
    with open(os.path.join(log_dir, "metrics2D.txt"), "a") as f:
        f.write(f"avg loss: {val_loss / val_num}, avg lpips: {val_lpips / val_num}, avg ssim: {val_ssim / val_num}, avg psnr: {val_psnr / val_num}")
        # add a new line
        f.write("\n")
