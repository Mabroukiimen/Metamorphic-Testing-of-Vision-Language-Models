from typing import Optional
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo


class LaMaInpainter:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        self.device = device

        train_config = OmegaConf.load(config_path)
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        self.model = load_checkpoint(
            train_config,
            checkpoint_path,
            strict=False,
            map_location=device,
        )
        self.model.freeze()
        self.model.to(device)

    def inpaint(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        img_bgr: HxWx3 uint8
        mask: HxW uint8, values 0/1 or 0/255
        returns: HxWx3 uint8 BGR
        """
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        mask = (mask > 0).astype(np.uint8)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        image_t = torch.from_numpy(img_rgb).float() / 255.0
        image_t = image_t.permute(2, 0, 1).unsqueeze(0).to(self.device)

        mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(self.device)

        batch = {
            "image": image_t,
            "mask": mask_t,
        }

        h, w = image_t.shape[2], image_t.shape[3]

        batch["image"] = pad_tensor_to_modulo(batch["image"], 8)
        batch["mask"] = pad_tensor_to_modulo(batch["mask"], 8)

        with torch.no_grad():
            batch = self.model(batch)
            result = batch["inpainted"][0].permute(1, 2, 0).cpu().numpy()

        result = result[:h, :w, :]
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)