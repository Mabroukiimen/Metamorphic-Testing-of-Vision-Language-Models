from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class Det:
    cls_id: int
    cls_name: str
    conf: float
    bbox_xyxy: Tuple[int, int, int, int]


class YOLODetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_thres: float = 0.25,
    ):
        self.conf_thres = conf_thres
        self.model = YOLO(model_path)

    def detect_topk(self, pil_img, topk: int = 10) -> List[Det]:
        """
        Run YOLO on a PIL image and return top-k detections sorted by confidence.
        """
        image_rgb = np.array(pil_img.convert("RGB"))

        results = self.model.predict(
            source=image_rgb,
            conf=self.conf_thres,
            verbose=False
        )

        if not results:
            return []

        r = results[0]
        if r.boxes is None:
            return []

        names = r.names
        dets: List[Det] = []

        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            cls_id = int(box.cls[0].cpu().item())
            conf = float(box.conf[0].cpu().item())

            dets.append(
                Det(
                    cls_id=cls_id,
                    cls_name=names[cls_id],
                    conf=conf,
                    bbox_xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                )
            )

        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets[:topk]