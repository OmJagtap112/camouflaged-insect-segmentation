import os
import csv
import torch
import argparse

from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import get_loader, test_dataset
from Src.utils.trainer import trainer
from Src.utils.loss import BCEDiceLoss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batchsize', type=int, default=36)
    parser.add_argument('--trainsize', type=int, default=352)

    parser.add_argument('--clip', type=float, default=0.5)

    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument(
        '--save_model',
        type=str,
        default='./Snapshot/2026-Syndata/'
    )

    # -----------------------------
    # Training Dataset
    # -----------------------------

    parser.add_argument(
        '--train_img_dir',
        type=str,
        default=r"C:\Users\omdja\Desktop\Dataset\Resize_cam_split\train\Images"
    )

    parser.add_argument(
        '--train_gt_dir',
        type=str,
        default=r"C:\Users\omdja\Desktop\Dataset\Resize_cam_split\train\Masks"
    )

    # -----------------------------
    # Validation Dataset
    # -----------------------------

    parser.add_argument(
        '--val_img_dir',
        type=str,
        default=r"C:\Users\omdja\Desktop\Dataset\Resize_cam_split\val\Images"
    )

    parser.add_argument(
        '--val_gt_dir',
        type=str,
        default=r"C:\Users\omdja\Desktop\Dataset\Resize_cam_split\val\Masks"
    )

    opt = parser.parse_args()

    # =====================================================
    # Device
    # =====================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using Device :", device)

    # =====================================================
    # Model
    # =====================================================

    

    model = SINet_ResNet50(channel=32).to(device)

    # =====================================================
    # Optimizer
    # =====================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=opt.lr,

        weight_decay=1e-4

    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=opt.epoch,

        eta_min=1e-6

    )

    criterion = BCEDiceLoss()

    # =====================================================
    # Loaders
    # =====================================================

    train_loader = get_loader(

        opt.train_img_dir,

        opt.train_gt_dir,

        batchsize=opt.batchsize,

        trainsize=opt.trainsize,

        shuffle=True,

        num_workers=8

    )

    val_loader = test_dataset(

        image_root=opt.val_img_dir,

        gt_root=opt.val_gt_dir,

        testsize=opt.trainsize

    )

    total_step = len(train_loader)

    # =====================================================
    # CSV
    # =====================================================

    os.makedirs(opt.save_model, exist_ok=True)

    csv_path = os.path.join(

        opt.save_model,

        "new_validation_results.csv"

    )

    csv_file = open(csv_path, "w", newline="")

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([

        "Epoch",

        "Train Loss",

        "Dice",

        "IoU",

        "Precision",

        "Recall"

    ])

    # =====================================================
    # Best Dice
    # =====================================================

    best_dice = 0

    # =====================================================
    # Training Loop
    # =====================================================

    print("=" * 70)
    print("Training Started")
    print("=" * 70)

    for epoch in range(1, opt.epoch + 1):

        best_dice = trainer(

            train_loader=train_loader,

            val_loader=val_loader,

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,

            epoch=epoch,

            opt=opt,

            loss_func=criterion,

            total_step=total_step,

            device=device,

            best_dice=best_dice,

            csv_writer=csv_writer

        )

    csv_file.close()

    print("\n")

    print("=" * 70)

    print("Training Finished")

    print("Best Validation Dice :", round(best_dice, 4))

    print("Validation CSV Saved :", csv_path)

    print("=" * 70)