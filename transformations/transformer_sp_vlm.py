# transformations/transformer_sp_vlm.py
from typing import Sequence
from PIL import Image, ImageEnhance, ImageFilter
from Utils.vector_layout import Vec

class SPTransformer:
    def __init__(self, img: Image.Image):
        self.img = img.convert("RGB")

    def apply_pixel(self, v: Sequence[float]) -> Image.Image:
        v = list(v)

        if round(v[Vec.B_BRIGHT]) == 1:
            self.img = ImageEnhance.Brightness(self.img).enhance(float(v[Vec.BRIGHT_FACTOR]))

        if round(v[Vec.B_BLUR]) == 1:
            self.img = self.img.filter(ImageFilter.GaussianBlur(radius=float(v[Vec.BLUR_RADIUS])))

        return self.img

    def apply_geometric(self, v: Sequence[float]) -> Image.Image:
        v = list(v)

        if round(v[Vec.B_ROTATE]) == 1:
            self.img = self.img.rotate(float(v[Vec.ROT_ANGLE]), resample=Image.BILINEAR, expand=False)

        if round(v[Vec.B_TRANSLATE]) == 1:
            tx = int(v[Vec.TX]); ty = int(v[Vec.TY])
            w, h = self.img.size
            canvas = Image.new("RGB", (w, h), (0, 0, 0))
            canvas.paste(self.img, (tx, ty))
            self.img = canvas

        return self.img