# transformations/transformer_sp_vlm.py
from typing import Sequence
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
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
        
        if round(v[Vec.B_CONTRAST]) == 1:
            self.img = ImageEnhance.Contrast(self.img).enhance(float(v[Vec.CONTRAST_FACTOR]))
        
        if round(v[Vec.B_SATURATION]) == 1:
            self.img = ImageEnhance.Color(self.img).enhance(float(v[Vec.SATURATION_FACTOR]))
            
        if round(v[Vec.B_NOISE]) == 1:
            std = float(v[Vec.NOISE_STD])
            arr = np.asarray(self.img).astype(np.float32)
            noise = np.random.normal(0.0, std, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            self.img = Image.fromarray(arr)

        return self.img

    def apply_geometric(self, v: Sequence[float]) -> Image.Image:
        v = list(v)

        if round(v[Vec.B_ROTATE]) == 1:
            self.img = self.img.rotate(float(v[Vec.ROT_ANGLE]), resample=Image.BILINEAR, expand=False)

        if round(v[Vec.B_TRANSLATE]) == 1:
            tx = int(round(v[Vec.TX])); ty = int(round(v[Vec.TY]))
            w, h = self.img.size
            canvas = Image.new("RGB", (w, h), (0, 0, 0))
            canvas.paste(self.img, (tx, ty))
            self.img = canvas
            
        if round(v[Vec.B_FLIP]) == 1:
            self.img = ImageOps.mirror(self.img)
            
        if round(v[Vec.B_SHEAR]) == 1:
            shear_x = float(v[Vec.SHEAR_X])
            shear_y = float(v[Vec.SHEAR_Y])
            w, h = self.img.size
            self.img = self.img.transform(
                (w, h),
                Image.AFFINE,
                (1.0, shear_x, 0.0, shear_y, 1.0, 0.0),
                resample=Image.BICUBIC
                )
            
        if round(v[Vec.B_ZOOM]) == 1:
            z = float(v[Vec.ZOOM_FACTOR])
            w, h = self.img.size
            
            if z > 1.0:
                nw, nh = int(w * z), int(h * z)
                zoomed = self.img.resize((nw, nh), Image.BICUBIC)
                left = (nw - w) // 2
                top = (nh - h) // 2
                self.img = zoomed.crop((left, top, left + w, top + h))
                
            elif z < 1.0:
                nw, nh = max(1, int(w * z)), max(1, int(h * z))
                zoomed = self.img.resize((nw, nh), Image.BICUBIC)
                canvas = Image.new("RGB", (w, h), (0, 0, 0))
                left = (w - nw) // 2
                top = (h - nh) // 2
                canvas.paste(zoomed, (left, top))
                self.img = canvas
                
        return self.img