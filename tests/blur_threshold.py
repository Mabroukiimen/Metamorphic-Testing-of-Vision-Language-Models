from PIL import Image, ImageFilter
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


def apply_blur(image_path, blur_radius, output_path=None):
    # Open original image
    original = Image.open(image_path).convert("RGB")

    # Apply Gaussian blur
    blurred_img = original.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Save if needed
    if output_path is not None:
        blurred_img.save(output_path)

    # Compute PSNR
    psnr_value = compute_psnr(original, blurred_img)

    return blurred_img, psnr_value


if __name__ == "__main__":
    image_path = "/home/ubuntu/Metamorphic-Testing-for-Vision-Language-Models/base_tests_images/000000581886.jpg"  
    blur_radius = 2.5                                      
    output_path = f"/home/ubuntu/last_version/tests/blur_manual_verification/581886/blur_img_{str(blur_radius).replace('.', '_')}.jpg"

    _, psnr = apply_blur(image_path, blur_radius, output_path)

    print(f"Blur radius: {blur_radius}")
    print(f"PSNR: {psnr}")