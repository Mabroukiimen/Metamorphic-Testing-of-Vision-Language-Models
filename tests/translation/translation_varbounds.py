from PIL import Image


def translate_image(image_path, tx, ty, output_path=None):
    image = Image.open(image_path).convert("RGB")

    # affine transform for translation
    translated_img = image.transform(
        image.size,
        Image.AFFINE,
        (1, 0, tx, 0, 1, ty)
    )

    if output_path is not None:
        translated_img.save(output_path)

    return translated_img


if __name__ == "__main__":
    image_path = "/home/ubuntu/Metamorphic-Testing-for-Vision-Language-Models/base_tests_images/000000581887.jpg"   # change this
    tx = 60    # shift in x
    ty = 60    # shift in y

    output_path = f"/home/ubuntu/last_version/tests/translation/translated_img_{tx}_{ty}.jpg"

    translate_image(image_path, tx, ty, output_path)

    print(f"Translation x: {tx}")
    print(f"Translation y: {ty}")
    print(f"Saved image: {output_path}")