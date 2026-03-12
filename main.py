import yaml, json, random
from pathlib import Path

import sys
sys.path.insert(0, "/home/ubuntu/lama")

from Models.sts_scorer import STSScorer
#from Models.llava_runner import VLMRunner
from Models.vlm_runner import VLMRunner
# from Models.blip_runner import BLIPRunner

from perception.yolo_detector import YOLODetector
from perception.sam_segmenter import SAMSegmenter

from optimization_algorithms.genetic_algorithm.geneticalgorithm import geneticalgorithm
from optimization_algorithms.ga_driver import run_ga

from perception.lama_inpainter import LaMaInpainter

def load_base_tests(meta_path: str):
    return json.loads(Path(meta_path).read_text(encoding="utf-8"))

def main(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).open("r", encoding="utf-8"))

    tests = load_base_tests(cfg["paths"]["base_images_meta"])
    sample = random.choice(tests)

    base_image_path = sample["image_path"]
    image_id = sample.get("image_id")

    sts = STSScorer(cfg["sts"]["model_name"])
    vlm = VLMRunner(
    model_name=cfg["vlm"]["model_name"],
    vlm_root=cfg["vlm"]["vlm_root"],
    python_cmd=cfg["vlm"]["python_cmd"],
    script_path=cfg["vlm"].get("script_path"),
)

    yolo = YOLODetector(model_path="yolov8l.pt", conf_thres=cfg["thresholds"].get("yolo_conf", 0.25))
    sam_segmenter = SAMSegmenter(
    checkpoint_path="/home/ubuntu/segment-anything/sam_vit_h_4b8939.pth",
    model_type="vit_h",
    device="cuda"
)
    
    lama_inpainter = LaMaInpainter(
    config_path="/home/ubuntu/lama/big-lama/config.yaml",
    checkpoint_path="/home/ubuntu/lama/big-lama/models/best.ckpt",
    device="cuda",
)

    run_ga(
        cfg=cfg,
        base_image_path=base_image_path,
        image_id=image_id,
        vlm_runner=vlm,
        sts_scorer=sts,
        yolo_detector=yolo,
        sam_segmenter=sam_segmenter,
        lama_inpainter=lama_inpainter,
    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)