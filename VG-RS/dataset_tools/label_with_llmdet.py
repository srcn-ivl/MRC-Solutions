import argparse
import json
import os
from typing import Any, Dict, List, Optional, Union
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import re
from tqdm import tqdm
from groundingdino.util.inference import load_model, predict
from PIL import Image

from qwen_vl_utils import smart_resize, process_vision_info
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from transformers.image_utils import load_image
def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Add candidates by Grounding DINO")
    parser.add_argument(
        "--anno_path",
        type=str,
        required=True,
        help="Path to input annotation file in JSONL format",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root directory containing the images",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the output annotation file",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Text threshold for Grounding DINO",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Box threshold for Grounding DINO",
    )
    return parser.parse_args()


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

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area


def return_maximum_overlap(gt_box, candidate_boxes, min_iou=0.5):
    max_iou = 0.0
    best_box = None
    for box in candidate_boxes:
        iou = compute_iou(gt_box, box)
        if iou >= min_iou and iou > max_iou:
            max_iou = iou
            best_box = box
    return best_box


def extract_bbox_answer(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        bbox_match = re.search(bbox_pattern, content_answer, re.DOTALL)
        if bbox_match:
            bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
            return bbox
    return [0, 0, 0, 0]

def gdino_load_image(image: Union[str, Image.Image]) -> torch.Tensor:
    """Load and transform image for Grounding DINO model.

    Args:
        image (Union[str, Image.Image]): Input image path or PIL Image.

    Returns:
        torch.Tensor: Transformed image tensor.
    """
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if isinstance(image, str):
        image_source = Image.open(image).convert("RGB")
    else:
        image_source = image
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image_transformed


def inference_gdino(
    image: Image.Image,
    prompts: List[str],
    gdino_model: Any,
    TEXT_TRESHOLD: float = 0.25,
    BOX_TRESHOLD: float = 0.35,
) -> torch.Tensor:
    """Process an image with Grounding DINO model to detect objects.

    Args:
        image (Image.Image): Input PIL image.
        prompts (List[str]): List of text prompts for object detection.
        gdino_model (Any): The Grounding DINO model instance.
        TEXT_TRESHOLD (float, optional): Text confidence threshold. Defaults to 0.25.
        BOX_TRESHOLD (float, optional): Box confidence threshold. Defaults to 0.35.

    Returns:
        torch.Tensor: Tensor containing detected object bounding boxes in format (x1, y1, x2, y2).
    """
    text_labels = ".".join(prompts)
    image_transformed = gdino_load_image(image)
    boxes, _, _ = predict(
        model=gdino_model,
        image=image_transformed,
        caption=text_labels,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
    )
    # the output boxes is in the format of (x,y,w,h), in [0,1]
    boxes = boxes * torch.tensor([image.width, image.height, image.width, image.height])
    # convert to the format of (x1,y1,x2,y2)
    boxes = torch.cat(
        (boxes[:, :2] - boxes[:, 2:4] / 2, boxes[:, :2] + boxes[:, 2:4] / 2), dim=1
    )
    return boxes.tolist()


def aux_resize_bbox(bbox, input_height, input_width, image_height, image_width):
    bbox[0] = bbox[0] / input_width * image_width
    bbox[1] = bbox[1] / input_height * image_height
    bbox[2] = bbox[2] / input_width * image_width
    bbox[3] = bbox[3] / input_height * image_height 
    return bbox

def add_candidate_boxes(
    anno_path: str,
    image_root: str,
    save_path: str,
    gdino_model: Any,
    TEXT_TRESHOLD: float = 0.25,
    BOX_TRESHOLD: float = 0.25,
) -> None:
    """Add candidate boxes to annotations using Grounding DINO model.

    This function processes a custom dataset for referring expression comprehension task in Rex-Thinker-GRPO.
    The input annotation file should be in JSONL format, where each line contains a JSON object with the following structure:

    Example input format:
    ```json
    {
        "image_name": "COCO_train2014_000000292799.jpg",
        "ans": [82.92, 201.37, 479.07, 586.99], # x0, y0, x1, y1
        "category": "turkey",
        "referring": "two turkeys near a tree"
    }
    ```

    The output will add a "candidate_boxes" field to each annotation, containing detected boxes from Grounding DINO.

    Args:
        anno_path (str): Path to input annotation file in JSONL format.
        image_root (str): Root directory containing the images.
        save_path (str): Path to save the output annotation file.
        gdino_model (Any): The Grounding DINO model instance.
        TEXT_TRESHOLD (float, optional): Text confidence threshold. Defaults to 0.25.
        BOX_TRESHOLD (float, optional): Box confidence threshold. Defaults to 0.35.
    """
    with open(anno_path, "r") as f:
        annos = [json.loads(line) for line in f]
        
    # aux_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "/data2/tangyin/models/r1/qwen_2.5_r1_rec_coco_2k4_bbox_refined/",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="cuda:0",
    # )
    model_id = "/data2/tangyin/models/llmdet/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


    for anno in tqdm(annos):
        image_path = os.path.join(image_root, anno["image"])
        # image = Image.open(image_path).convert("RGB")
            # Prepare inputs
        image = load_image(image_path)
        text_labels = [[anno['category']]]
        # print(text_labels)
        try:
            inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
        
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)

            # Postprocess outputs
            results = processor.post_process_grounded_object_detection(
                outputs,
                threshold=0.25,
                target_sizes=[(image.height, image.width)]
            )

            # Retrieve the first image result
            result = results[0]
            boxes = []
            for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
                box = [round(x, 2) for x in box.tolist()]
                boxes.append(box)
                # print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
        except:
            boxes = []
            
        ref_exp = anno["problem"].replace("Please provide the bounding box coordinate of the region this sentence describes:", "").strip().replace(".", "")
            
        resized_boxes = convert_boxes_from_absolute_to_qwen25_format(boxes, image.width, image.height)
        category = anno["category"]
        
        hint = json.dumps(
            {
                f"{category}": resized_boxes,
            }, ensure_ascii=False
        )
        question = f"Hint: Object and its coordinates in this image: {hint}\nPlease detect {ref_exp} in the image."
        # we need to make sure that the gt_box is inside the candidate boxes
        # best_box = return_maximum_overlap(gt_box, boxes)
        # if best_box is None:
        #     boxes.append(gt_box)
        
        anno['problem'] = question
        del anno['category']
    with open(save_path, "w", encoding='utf-8') as f:
        for anno in annos:
            json.dump(anno, f, ensure_ascii=False)
            f.write("\n")


def main() -> None:
    """Main function to run the script."""
    args = parse_args()
    gdino_model = load_model(
        args.gdino_config,
        args.gdino_weights,
    ).to("cuda:2")
    add_candidate_boxes(
        args.anno_path,
        args.image_root,
        args.save_path,
        gdino_model,
        args.text_threshold,
        args.box_threshold,
    )


if __name__ == "__main__":
    main()
