import numpy as np
from PIL import Image

def compute_psnr_pil(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(img1).astype(np.float32)
    b = np.array(img2).astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))   #10.0 * np.log10((max_pixel ** 2) / mse)