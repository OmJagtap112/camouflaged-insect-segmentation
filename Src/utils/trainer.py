import os
import csv
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from datetime import datetime
from torch.autograd import Variable


# ==========================================================
# Validation
# ==========================================================

def validate(model, val_loader, device):

    model.eval()

    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []

    with torch.no_grad():

        val_loader.index = 0

        for _ in range(val_loader.size):

            image, gt, _ = val_loader.load_data()

            image = image.to(device)

            gt = np.array(gt, np.float32)

            _, pred = model(image)

            pred = F.interpolate(
                pred,
                size=gt.shape,
                mode="bilinear",
                align_corners=False
            )

            pred = torch.sigmoid(pred)

            pred = pred.cpu().numpy().squeeze()

            pred = (pred - pred.min()) / (
                pred.max() - pred.min() + 1e-8
            )

            pred = (pred > 0.5).astype(np.uint8)
            gt = (gt > 127).astype(np.uint8)

            tp = np.sum((pred == 1) & (gt == 1))
            fp = np.sum((pred == 1) & (gt == 0))
            fn = np.sum((pred == 0) & (gt == 1))

            dice = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)
            iou = (tp + 1e-8) / (tp + fp + fn + 1e-8)
            precision = (tp + 1e-8) / (tp + fp + 1e-8)
            recall = (tp + 1e-8) / (tp + fn + 1e-8)

            dice_scores.append(dice)
            iou_scores.append(iou)
            precision_scores.append(precision)
            recall_scores.append(recall)

    model.train()

    return (
        np.mean(dice_scores),
        np.mean(iou_scores),
        np.mean(precision_scores),
        np.mean(recall_scores)
    )


# ==========================================================
# Trainer
# ==========================================================

def trainer(
        train_loader,
        val_loader,
        model,
        optimizer,
        scheduler,
        epoch,
        opt,
        loss_func,
        total_step,
        device,
        best_dice,
        csv_writer
):

    model.train()

    epoch_loss = 0.0

    print("\n" + "=" * 70)
    print(f"Epoch {epoch}/{opt.epoch}")
    print("=" * 70)

    for step, data_pack in enumerate(train_loader):

        images, gts = data_pack

        images = Variable(images).to(device)
        gts = Variable(gts).to(device)

        optimizer.zero_grad()

        cam_sm, cam_im = model(images)

        loss_sm = loss_func(cam_sm, gts)
        loss_im = loss_func(cam_im, gts)

        loss = loss_sm + loss_im

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            opt.clip
        )

        optimizer.step()

        epoch_loss += loss.item()

        if step % 10 == 0:

            print(
                "[{}] Epoch {:03d}/{:03d} "
                "Step {:04d}/{:04d} "
                "Loss {:.4f}".format(
                    datetime.now().strftime("%H:%M:%S"),
                    epoch,
                    opt.epoch,
                    step,
                    total_step,
                    loss.item()
                )
            )

    scheduler.step()

    avg_loss = epoch_loss / total_step

    # ======================================================
    # Validation
    # ======================================================

    dice, iou, precision, recall = validate(
        model,
        val_loader,
        device
    )

    print("\nValidation Results")

    print(f"Dice      : {dice:.4f}")
    print(f"IoU       : {iou:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")

    # ======================================================
    # CSV Logging
    # ======================================================

    csv_writer.writerow([
        epoch,
        avg_loss,
        dice,
        iou,
        precision,
        recall
    ])

    # ======================================================
    # Save Every Epoch
    # ======================================================

    os.makedirs(opt.save_model, exist_ok=True)

    epoch_model = os.path.join(
        opt.save_model,
        f"SINet_{epoch}.pth"
    )

    torch.save(
        model.state_dict(),
        epoch_model
    )

    print(f"\nSaved : {epoch_model}")

    # ======================================================
    # Save Best Model
    # ======================================================

    if dice > best_dice:

        best_dice = dice

        best_model = os.path.join(
            opt.save_model,
            "best_model.pth"
        )

        torch.save(
            model.state_dict(),
            best_model
        )

        print("\n★★★★★ BEST MODEL UPDATED ★★★★★")
        print(f"Validation Dice : {best_dice:.4f}")

    print(f"\nLearning Rate : {scheduler.get_last_lr()[0]:.8f}")

    return best_dice