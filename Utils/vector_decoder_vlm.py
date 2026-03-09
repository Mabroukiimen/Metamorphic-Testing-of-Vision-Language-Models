from typing import Sequence
from PIL import Image
from transformations.transformer_sp_vlm import SPTransformer
from transformations.transformer_sa_vlm import apply_sa

class VectorDecoderVLM:
    def __init__(self, image: Image.Image):
        self.image = image

    def apply_pixel(self, tr_vector: Sequence[float]) -> Image.Image:
        tr = SPTransformer(self.image.copy())
        return tr.apply_pixel(tr_vector)

    def apply_geometric(self, img_after_pixel: Image.Image, tr_vector: Sequence[float]) -> Image.Image:
        tr = SPTransformer(img_after_pixel.copy())
        return tr.apply_geometric(tr_vector)