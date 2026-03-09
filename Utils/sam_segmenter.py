import numpy as np
import cv2

class SAMSegmenter:
    def __init__(self, predictor):
        self.predictor = predictor

    def mask_from_box(self, img_pil, bbox_xyxy):
        # PIL -> numpy RGB
        image_rgb = np.array(img_pil.convert("RGB"))

        # tell SAM which image to work on
        self.predictor.set_image(image_rgb)

        # box format: [x1, y1, x2, y2]
        input_box = np.array(bbox_xyxy, dtype=np.float32)

        # gives SAM the predicted masks
        masks, scores, _ = self.predictor.predict(
            box=input_box,
            multimask_output=True
        )

        # keep the best mask
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8)

        return mask