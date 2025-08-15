# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import tempfile
from PIL import Image, ImageDraw
import time
import warnings

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "./")  # noqa
from demo.predictors import OVDINODemo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.utils.logger import setup_logger
from detrex.data.datasets import clean_words_or_phrase

from qwen_vl_utils import smart_resize

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    SAM2_AVAILABLE = True
except ImportError:
    print("SAM2 is not installed.")
    SAM2_AVAILABLE = False

# constants
WINDOW_NAME = "COCO detections"


def setup(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="detrex demo for visualizing customized inputs"
    )
    parser.add_argument(
        "--config-file",
        default="/home/pengziyang/MultimodalReasoningCompetition/Task1/models/OV-DINO/ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_infer_demo.py",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--sam-config-file",
        default=None,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--sam-init-checkpoint",
        default=None,
        metavar="FILE",
        help="path to sam checkpoint file",
    )
    parser.add_argument(
        "--min_size_test",
        type=int,
        default=800,
        help="Size of the smallest side of the image during testing. Set to zero to disable resize in testing.",
    )
    parser.add_argument(
        "--max_size_test",
        type=float,
        default=1333,
        help="Maximum size of the side of the image during testing.",
    )
    parser.add_argument(
        "--img_format",
        type=str,
        default="RGB",
        help="The format of the loading images.",
    )
    parser.add_argument(
        "--metadata_dataset",
        type=str,
        default="coco_2017_val",
        help="The metadata infomation to be used. Default to COCO val metadata.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def convert_boxes_from_absolute_to_qwen25_format(gt_boxes, ori_width, ori_height):
    """Convert bounding boxes from absolute coordinates to Qwen-25 format.

    This function resizes bounding boxes according to Qwen-25's requirements while
    maintaining aspect ratio and pixel constraints.

    Args:
        gt_boxes (List[List[float]]): List of bounding boxes in absolute coordinates.
        ori_width (int): Original image width.
        ori_height (int): Original image height.

    Returns:
        List[List[int]]: Resized bounding boxes in Qwen-25 format.
    """
    resized_height, resized_width = smart_resize(
        ori_height,
        ori_width,
        28,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    resized_gt_boxes = []
    for box in gt_boxes:
        # resize the box
        x0, y0, x1, y1 = box
        x0 = int(x0 / ori_width * resized_width)
        x1 = int(x1 / ori_width * resized_width)
        y0 = int(y0 / ori_height * resized_height)
        y1 = int(y1 / ori_height * resized_height)

        x0 = max(0, min(x0, resized_width - 1))
        y0 = max(0, min(y0, resized_height - 1))
        x1 = max(0, min(x1, resized_width - 1))
        y1 = max(0, min(y1, resized_height - 1))
        resized_gt_boxes.append([x0, y0, x1, y1])
        
    return resized_gt_boxes


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    # load vgrs dataset
    with open("/home/pengziyang/MultimodalReasoningCompetition/Task1/datasets/VG_RS/VGRS_WITH_CATE_CN.jsonl", 'r', encoding='utf-8') as f:
        annos = [json.loads(line) for line in f]
        
    cfg = setup(args)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.train.init_checkpoint)

    model.eval()
    demo = OVDINODemo(
        model=model,
        sam_predictor=None,
        min_size_test=args.min_size_test,
        max_size_test=args.max_size_test,
        img_format=args.img_format,
        metadata_dataset=args.metadata_dataset,
    )
    for anno in tqdm(annos):
        
        image_path = os.path.join("/home/pengziyang/MultimodalReasoningCompetition/Task1/datasets/VG_RS", anno['image'])
        image_RGB = Image.open(image_path)
        # use PIL, to be consistent with evaluation
        img = read_image(image_path, format="BGR")
        start_time = time.time()
        # print(f"category:````````````{anno['category']}")
        predictions = demo.run_on_image(
            img, [anno['category']], args.confidence_threshold
        )
        # print(predictions)
        json_results = instances_to_coco_json(
            predictions["instances"].to(demo.cpu_device), 0
        )
        
        boxes = []
        ref_exp = anno["problem"].replace("Please provide the bounding box coordinate of the region this sentence describes:", "").strip().replace(".", "")
        for json_result in json_results:
            json_result["category_name"] = anno['category']
            xywh = json_result['bbox']
            x1,y1 = xywh[0], xywh[1]
            x2 = x1 + xywh[2]
            y2 = y1 + xywh[3]
            boxes.append([x1,y1,x2,y2])
            del json_result["image_id"]

        # logger.info(
        #     "{}: {} in {:.2f}s".format(
        #         image_path,
        #         (
        #             "detected {} instances".format(len(predictions["instances"]))
        #             if "instances" in predictions
        #             else "finished"
        #         ),
        #         time.time() - start_time,
        #     )
        # )

        resized_boxes = convert_boxes_from_absolute_to_qwen25_format(boxes, image_RGB.width, image_RGB.height)
        category = anno["category"]
        
        hint = json.dumps(
            {
                f"{category}": resized_boxes,
            }, ensure_ascii=False
        )
        question = f"Hint: Object and its coordinates in this image: {hint}\nPlease detect {ref_exp} in the image."
        anno['problem'] = question
        del anno['category']
        # del demo
        
    save_path = "/home/pengziyang/MultimodalReasoningCompetition/Task1/datasets/VG_RS/ovdino_label/0804_ovdino_candidate_box_iou30.jsonl"
    with open(save_path, "w", encoding='utf-8') as f:
        for anno in annos:
            json.dump(anno, f, ensure_ascii=False)
            f.write("\n")
