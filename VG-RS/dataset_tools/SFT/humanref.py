from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import torch, cv2, argparse
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np
from datetime import datetime
import os
import time
import json
from rich.progress import track
import logging
# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import csv
from io import BytesIO
data = []
from base64 import b64decode
ann_lineidx = open("/data2/tangyin/dataset/humanref/humanref_cot.annotations.tsv.lineidx")
for line in ann_lineidx:
    data.append(int(line.strip()))

SFT_DATA = []
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("/home/tangyin/demo_poc/qwen2.5_vl/", min_pixels=min_pixels, max_pixels=max_pixels)

def resize_bbox(bbox, input_height, input_width, image_height, image_width):
    bbox[0] = bbox[0] / image_width * input_width
    bbox[1] = bbox[1] / image_height * input_height 
    bbox[2] = bbox[2] / image_width * input_width
    bbox[3] = bbox[3] / image_height * input_height 
    return bbox

for idx in track(data):

    ann_handle = open("/data2/tangyin/dataset/humanref/humanref_cot.annotations.tsv")
    ann_handle.seek(idx)

    image_line_idx, ann = ann_handle.readline().strip().split("\t")
    image_path = os.path.join("/data2/tangyin/dataset/humanref/images/", image_line_idx + '.jpg')
    image_line_idx = int(image_line_idx)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_path}",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_height = int(inputs['image_grid_thw'][0][1]*14)
    input_width = int(inputs['image_grid_thw'][0][2]*14)
    raw_image = Image.open(image_path)
    image_height = raw_image.height
    image_width = raw_image.width
    # print(f"input h w :{input_height} {input_width} - {image_line_idx}")
    del raw_image
    ann = eval(ann)
    
    gt_bboxs = ann['gt_boxes']
    region_map = ann['region_map']
    think = ann['think']
    
    # regular gt bbox align with input size
    for idx, box in enumerate(gt_bboxs):
        x1, y1, w, h = box[0], box[1], box[2], box[3]
        x2 = x1 + w
        y2 = y1 + h
        temp = [x1, y1, x2, y2]
        resized_bbox = resize_bbox(temp, input_height,input_width, image_height, image_width)
        resized_bbox[0] = round(resized_bbox[0])
        resized_bbox[1] = round(resized_bbox[1])
        resized_bbox[2] = round(resized_bbox[2])
        resized_bbox[3] = round(resized_bbox[3])
        gt_bboxs[idx] = resized_bbox
        
    detected_label = []
    reference = None
    for ref, cords in region_map.items():
        reference = ref
        for cord in cords:
            detected_label.append(
                dict(
                    bbox_2d = gt_bboxs[cord],
                    label = ref
                )
            )
    hint = json.dumps(
        {
            "{cate_name}".format(cate_name='people'): gt_bboxs,
        }
    )
    # print(hint)
    question = f"Hint: Object and its coordinates in this image: {hint}\nPlease detect {reference} in the image."
    detected_label = str(detected_label)
    model_response = f"<think>\n{think}\n</think>\n<answer>\n{detected_label}\n</answer>"
    SYSTEM_PROMPT = "First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    curr_item = {
        "image": f"{image_path}",
        "conversations": [
            {
                "from": "human",
                # "value": f"<image>\nPlease provide the bounding box coordinate of the region this sentence describes:{reference}. {SYSTEM_PROMPT}"
                "value": f"<image>\n{question}"
            },
            {
                "from": "gpt",
                "value": f"{model_response}"
            }
        ]
    }
    SFT_DATA.append(curr_item)

with open(os.path.join("{Huamnref_SFT}.json".format(Huamnref_SFT = "/data2/tangyin/dataset/humanref/humanref_sft_1280px_with_hint")), 'w', encoding='utf-8') as f:
    json.dump(SFT_DATA, f, indent=4, ensure_ascii=False)
    