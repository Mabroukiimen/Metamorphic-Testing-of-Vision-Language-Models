from pathlib import Path
from PIL import Image

from transformations.transformer_sa_vlm import apply_sa
from perception.yolo_detector import YOLODetector
from perception.sam_segmenter import SAMSegmenter
from perception.lama_inpainter import LaMaInpainter


def main():
    image_path = "/home/ubuntu/Metamorphic-Testing-for-Vision-Language-Models/base_tests_images/000000581886.jpg"
    object_corpus_dir = "/home/ubuntu/segment-anything/object_corpus"

    sa_type = 5
    target_det_idx = 0
    obj_scale_factor = 1.3

    img = Image.open(image_path).convert("RGB")

    yolo = YOLODetector()
    sam = SAMSegmenter()
    lama = LaMaInpainter(
        config_path="/home/ubuntu/lama/big-lama/config.yaml",
        checkpoint_path="/home/ubuntu/lama/big-lama/models/best.ckpt",
        device="cuda",
    )

    dets = yolo.detect_topk(img, topk=10)

    if len(dets) == 0:
        raise ValueError("No detections found in the image.")

    if target_det_idx >= len(dets):
        raise ValueError(f"target_det_idx={target_det_idx} but only {len(dets)} detections were found.")

    print("Chosen class:", dets[target_det_idx].cls_name)
    print("Chosen bbox:", dets[target_det_idx].bbox_xyxy)

    final_img, sa_log = apply_sa(
        img_sp=img,
        sa_type=sa_type,
        tr_vector=None,
        yolo_dets=dets,
        chosen_idx=target_det_idx,
        sam_segmenter=sam,
        lama_inpainter=lama,
        object_corpus_dir=object_corpus_dir,
        ins_corpus_id=None,
        ins_scale=None,
        rep_corpus_id=None,
        rep_scale=None,
        obj_scale_factor=obj_scale_factor,
    )

    print(sa_log)

    out_path = Path("/home/ubuntu/last_version/tests/test_scale_object_output.png")
    final_img.save(out_path)
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()