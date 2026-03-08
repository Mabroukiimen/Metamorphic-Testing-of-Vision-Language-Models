from PIL import Image


def rotate_image(image_path, alpha, output_path=None):
    image = Image.open(image_path).convert("RGB")

    rotated_img = image.rotate(alpha, expand=False)

    if output_path is not None:
        rotated_img.save(output_path)

    return rotated_img


if __name__ == "__main__":
    image_path = "/home/ubuntu/Metamorphic-Testing-for-Vision-Language-Models/base_tests_images/000000581887.jpg"   
    alpha = -45                                         # rotation angle in degrees
    output_path = f"/home/ubuntu/last_version/tests/rotation/rotated_img_{alpha}.jpg"

    rotate_image(image_path, alpha, output_path)

    print(f"Rotation angle: {alpha}")
    print(f"Saved image: {output_path}")