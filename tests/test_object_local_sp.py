from pathlib import Path
from PIL import Image

from transformations.transformer_sa_vlm import apply_sa
from perception.yolo_detector import YOLODetector
from perception.sam_segmenter import SAMSegmenter
from Utils.vector_layout import Vec


def make_zero_vector():
    return [0.0] * Vec.N


def run_case(case_name, image_path, target_det_idx, setup_vector_fn):
    img = Image.open(image_path).convert("RGB")

    yolo = YOLODetector()
    sam = SAMSegmenter(
    checkpoint_path="/home/ubuntu/segment-anything/sam_vit_h_4b8939.pth",
    model_type="vit_h",
    device="cuda"
)

    dets = yolo.detect_topk(img, topk=10)
    if len(dets) == 0:
        raise ValueError("No detections found.")
    if target_det_idx >= len(dets):
        raise ValueError(f"target_det_idx={target_det_idx} but only {len(dets)} detections found.")

    print(f"\n--- {case_name} ---")
    print("chosen class:", dets[target_det_idx].cls_name)
    print("chosen bbox:", dets[target_det_idx].bbox_xyxy)

    v = make_zero_vector()
    v[Vec.SA_TYPE] = 4
    setup_vector_fn(v)

    final_img, sa_log = apply_sa(
        img_sp=img,
        sa_type=4,
        tr_vector=v,
        yolo_dets=dets,
        chosen_idx=target_det_idx,
        sam_segmenter=sam,
        lama_inpainter=None,
        object_corpus_dir="unused",
        ins_corpus_id=None,
        ins_scale=None,
        rep_corpus_id=None,
        rep_scale=None,
        obj_scale_factor=None,
    )

    print("sa_log:", sa_log)

    out_dir = Path("/home/ubuntu/last_version/tests/object_local_sp_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if final_img is None:
        print("Result: REJECTED")
    else:
        out_path = out_dir / f"{case_name}.png"
        final_img.save(out_path)
        print("Saved:", out_path)


def main():
    image_path = "/home/ubuntu/Metamorphic-Testing-for-Vision-Language-Models/base_tests_images/000000581886.jpg"
    target_det_idx = 0

    # contrast only
    run_case(
        "contrast_only",
        image_path,
        target_det_idx,
        lambda v: (
            v.__setitem__(Vec.B_CONTRAST, 1),
            v.__setitem__(Vec.CONTRAST_FACTOR, 4.0),
        ),
    )

    # noise only
    run_case(
        "noise_only",
        image_path,
        target_det_idx,
        lambda v: (
            v.__setitem__(Vec.B_NOISE, 1),
            v.__setitem__(Vec.NOISE_STD, 40.0),
        ),
    )

    # flip only
    run_case(
        "flip_only",
        image_path,
        target_det_idx,
        lambda v: (
            v.__setitem__(Vec.B_FLIP, 1),
        ),
    )


if __name__ == "__main__":
    main()