import torch
import torch.nn.functional as F
import numpy as np
import os
import imageio

from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset
#from Src.utils.trainer import eval_mae

# ==========================================
# SETTINGS
# ==========================================

TEST_SIZE = 352

MODEL_PATH = r'./Snapshot/2026-SINet-Synthetic data/best_model.pth'

IMAGE_ROOT = r'C:\Users\omdja\Desktop\Dataset\red_forest\test\Img\\'
GT_ROOT = r'C:\Users\omdja\Desktop\Dataset\red_forest\test\GT\\'

SAVE_PATH = r'./Result/CamTest/'

os.makedirs(SAVE_PATH, exist_ok=True)

# ==========================================
# DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print("Using Device:", device)
print("Model:", MODEL_PATH)
print("=" * 50)

# ==========================================
# LOAD MODEL
# ==========================================

model = SINet_ResNet50().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model Loaded Successfully!")

# ==========================================
# LOAD TEST DATA
# ==========================================

test_loader = test_dataset(
    image_root=IMAGE_ROOT,
    gt_root=GT_ROOT,
    testsize=TEST_SIZE
)

print("Total Test Images:", test_loader.size)

# ==========================================
# TESTING
# ==========================================

mae_list = []

for i in range(test_loader.size):

    image, gt, name = test_loader.load_data()

    gt = np.asarray(gt, np.float32)
    gt = gt / (gt.max() + 1e-8)

    image = image.to(device)

    with torch.no_grad():
        _, pred = model(image)

    pred = F.interpolate(
        pred,
        size=gt.shape,
        mode='bilinear',
        align_corners=True
    )

    pred = pred.sigmoid().cpu().numpy().squeeze()

    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    save_file = os.path.join(SAVE_PATH, name)

    imageio.imwrite(
        save_file,
        (pred * 255).astype(np.uint8)
    )

    print("Saved:", save_file)

    mae = np.mean(np.abs(pred - gt))
    mae_list.append(mae)

    print(
        f'[{i+1}/{test_loader.size}] '
        f'{name} | MAE = {mae:.6f}'
    )

# ==========================================
# FINAL RESULT
# ==========================================

print("\n" + "=" * 50)
print("Testing Completed Successfully!")
print("Average MAE:", np.mean(mae_list))
print("Prediction Masks Saved To:")
print(SAVE_PATH)
print("=" * 50)