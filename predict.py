import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from Src.SINet import SINet_ResNet50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SINetPredictor:

    def __init__(self,
                 weight_path="Snapshot/2026-Syndata/best_model.pth"):

        self.model = SINet_ResNet50()

        self.model.load_state_dict(
            torch.load(weight_path, map_location=DEVICE)
        )

        self.model.to(DEVICE)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((352,352)),
            transforms.ToTensor(),  
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def predict(self,image):

        original = np.array(image)

        h,w = original.shape[:2]

        image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            _, pred = self.model(tensor)

            pred = torch.sigmoid(pred)

            pred = F.interpolate(
                pred,
                size=(h,w),
                mode="bilinear",
                align_corners=False
            )

        mask = pred.squeeze().cpu().numpy()

        mask = (mask-mask.min())/(mask.max()-mask.min()+1e-8)

        mask_uint8=(mask*255).astype(np.uint8)
        _, mask_uint8=cv2.threshold(mask_uint8,128,255,cv2.THRESH_BINARY)

        overlay=original.copy()
        green=np.zeros_like(original)

        green[:,:,1]=mask_uint8

        overlay =cv2.addWeighted(original,0.7,green,0.3,0)

        return original,mask_uint8,overlay