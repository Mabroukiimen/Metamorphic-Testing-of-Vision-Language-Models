from PIL import Image, ImageEnhance
import numpy as np
import math


def compute_psnr(img1, img2):
    arr1 = np.asarray(img1).astype(np.float32)
    arr2 = np.asarray(img2).astype(np.float32)

    mse = np.mean((arr1 - arr2) ** 2)

    if mse == 0:
        return float("inf")

    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def apply_brightness(image_path, brightness_factor, output_path=None):
    # Open original image
    original = Image.open(image_path).convert("RGB")

    # Apply brightness
    enhancer = ImageEnhance.Brightness(original)
    bright_img = enhancer.enhance(brightness_factor)

    # Save if needed
    if output_path is not None:
        bright_img.save(output_path)

    # Compute PSNR
    psnr_value = compute_psnr(original, bright_img)

    return bright_img, psnr_value


if __name__ == "__main__":
    brightness_factor = 0.8  # 1.0 = original, >1 brighter, <1 darker
    image_path = "/home/ubuntu/Metamorphic-Testing-for-Vision-Language-Models/base_tests_images/000000581887.jpg"      
    output_path = f"/home/ubuntu/last_version/tests/581887/bright_img_{str(brightness_factor).replace('.', '_')}.jpg"
    
    _, psnr = apply_brightness(image_path, brightness_factor, output_path)

    print(f"Brightness factor: {brightness_factor}")
    print(f"PSNR: {psnr}")