from typing import Tuple
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


class SAMSegmenter:
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_h",
        device: str = "cuda",
    ):
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def mask_from_box(self, pil_img, bbox_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Return a binary mask (H, W) for the object inside bbox.
        Policy: choose the highest-score SAM mask.
        """
        image_rgb = np.array(pil_img.convert("RGB"))
        self.predictor.set_image(image_rgb)

        input_box = np.array(bbox_xyxy, dtype=np.float32)

        masks, scores, _ = self.predictor.predict(
            box=input_box,
            multimask_output=True
        )

        best_idx = int(np.argmax(scores))
        mask = (masks[best_idx] > 0).astype(np.uint8)
        return mask
    
    
    
    