from PIL import Image
from transformations.transformer_sp_vlm import SPTransformer
from Utils.vector_layout import Vec

def make_zero_vector():
    return [0.0] * Vec.N

img_path = "/home/ubuntu/Metamorphic-Testing-for-Vision-Language-Models/base_tests_images/000000581886.jpg"
img = Image.open(img_path).convert("RGB")

# -------- contrast test --------
v = make_zero_vector()
v[Vec.B_CONTRAST] = 1
v[Vec.CONTRAST_FACTOR] = 1.6

tr = SPTransformer(img.copy())
out = tr.apply_pixel(v)
out.save("/home/ubuntu/last_version/tests/sp_tests/test_contrast.png")

# -------- saturation test --------
v = make_zero_vector()
v[Vec.B_SATURATION] = 1
v[Vec.SATURATION_FACTOR] = 1.8

tr = SPTransformer(img.copy())
out = tr.apply_pixel(v)
out.save("/home/ubuntu/last_version/tests/sp_tests/test_saturation.png")

# -------- noise test --------
v = make_zero_vector()
v[Vec.B_NOISE] = 1
v[Vec.NOISE_STD] = 15.0

tr = SPTransformer(img.copy())
out = tr.apply_pixel(v)
out.save("/home/ubuntu/last_version/tests/sp_tests/test_noise.png")

# -------- flip test --------
v = make_zero_vector()
v[Vec.B_FLIP] = 1

tr = SPTransformer(img.copy())
out = tr.apply_geometric(v)
out.save("/home/ubuntu/last_version/tests/sp_tests/test_flip.png")

# -------- shear test --------
v = make_zero_vector()
v[Vec.B_SHEAR] = 1
v[Vec.SHEAR_X] = 0.2
v[Vec.SHEAR_Y] = 0.0

tr = SPTransformer(img.copy())
out = tr.apply_geometric(v)
out.save("/home/ubuntu/last_version/tests/sp_tests/test_shear.png")

# -------- zoom in test --------
v = make_zero_vector()
v[Vec.B_ZOOM] = 1
v[Vec.ZOOM_FACTOR] = 1.2

tr = SPTransformer(img.copy())
out = tr.apply_geometric(v)
out.save("/home/ubuntu/last_version/tests/sp_tests/test_zoom_in.png")

# -------- zoom out test --------
v = make_zero_vector()
v[Vec.B_ZOOM] = 1
v[Vec.ZOOM_FACTOR] = 0.8

tr = SPTransformer(img.copy())
out = tr.apply_geometric(v)
out.save("/home/ubuntu/last_version/tests/sp_tests/test_zoom_out.png")

print("Done")