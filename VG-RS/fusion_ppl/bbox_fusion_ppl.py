from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset, concatenate_datasets
from PIL import Image
from tqdm import tqdm
import torch, cv2, argparse
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np
from datetime import datetime
import os
import time
import json
import logging
from qwen_vl_utils import smart_resize

from ensemble_boxes import weighted_boxes_fusion


from torchvision.ops import box_convert
import matplotlib.pyplot as plt  
import matplotlib.patches as patches
import math
import re
def plot_bbox(image, data, label):
    # Create a figure and axes  
    fig, ax = plt.subplots()
    # Display the image  
    ax.imshow(image)
    # Plot each bounding box  

    # Unpack the bounding box coordinates  
    x1,y1,x2,y2 = data[0],data[1],data[2],data[3],

    # cropped_img = crop_image("/home/tangyin/images/samsungs24.jpg", math.ceil(x1) , math.ceil(y1), math.ceil(x2), math.ceil(y2), "output.jpg")
    # Create a Rectangle patch  
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none', linestyle="--")  
    # Add the rectangle to the Axes  
    ax.add_patch(rect)  
    # Annotate the label  
    plt.text(x1, y1, f'{label}', color='black', fontsize=15, bbox=dict(facecolor='pink', alpha=0.5))  
        
    # Remove the axis ticks and labels  
    ax.axis('off')  

    # Show the plot  
    plt.show()

def abs2propo(box, image_height, image_width):
    norm_box = [0,0,0,0]
    norm_box[0] = round(box[0] / image_width, 3)
    norm_box[1] = round(box[1] / image_height, 3)
    norm_box[2] = round(box[2] / image_width, 3)
    norm_box[3] = round(box[3] / image_height, 3)
    return norm_box

def propo2abs(box, image_height, image_width):
    norm_box = [0,0,0,0]
    norm_box[0] = box[0] * image_width
    norm_box[1] = box[1] * image_height
    norm_box[2] = box[2] * image_width
    norm_box[3] = box[3] * image_height
    return norm_box


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

def parse_json(json_output):
    pattern = r"\[([0-9\.]+(?:, ?[0-9\.]+)*)\]"
    matches = re.findall(pattern, json_output)
    coordinates = [
        [float(num) if "." in num else int(num) for num in match.split(",")]
        for match in matches
    ]
    return coordinates


def resize2image(bbox, input_height, input_width, image_height, image_width):
    new_box = [0,0,0,0]
    new_box[0] = min((bbox[0] / input_width * image_width), image_width)
    new_box[1] = min((bbox[1] / input_height * image_height), image_height)
    new_box[2] = min((bbox[2] / input_width * image_width), image_width)
    new_box[3] = min((bbox[3] / input_height * image_height), image_height)
    return new_box

def resize2input(bbox, input_height, input_width, image_height, image_width):
    new_box = [0,0,0,0]
    new_box[0] = min(round(bbox[0] / image_width * input_width), input_width)
    new_box[1] = min(round(bbox[1] / image_height * input_height), input_height)
    new_box[2] = min(round(bbox[2] / image_width * input_width), input_width)
    new_box[3] = min(round(bbox[3] / image_height * input_height), input_height)
    return new_box

with open("/home/pengziyang/MultimodalReasoningCompetition/Task1/datasets/VG_RS/0801_72B_2560.json", 'r', encoding='utf-8') as f:
    anno_72b = json.load(f)

with open("/home/pengziyang/MultimodalReasoningCompetition/Task1/datasets/VG_RS/ovdino_label/0805_candidate_ovdino_llmdet.jsonl", 'r', encoding='utf-8') as f:
    anno_candi = [json.loads(line) for line in f]
    
with open("/home/pengziyang/MultimodalReasoningCompetition/Task1/datasets/VG_RS/0730_GLM_RP_w_72B.json", 'r', encoding="utf-8") as f:
    anno_glm = json.load(f)
    
cnt = 0
cnt_gdino_replace = 0
cnt_gdino_fusion = 0
cnt_glm_replace = 0
cnt_glm_fusion = 0

iou_thr = 0.25
skip_box_thr = 0.0001
sigma = 0.1

image_save_path = "/home/pengziyang/MultimodalReasoningCompetition/Task1/datasets/VG_RS/visualize/"

edit_collect = []
glm_collect = []

glm_replace_cnt = 0

for anno72b in tqdm(anno_72b):
    image_key = anno72b['image_path'].split("\\")[1].strip()
    text_key = anno72b['question'].replace(".","").strip()
    for annocandi in anno_candi:
        image_path = annocandi['image'].split("/")[1].strip()
        text = annocandi['problem'].split("\n")[-1].replace("Please detect", "").replace("in the image.", "").strip()
        if image_path == image_key and text == text_key:

            image = Image.open(f"/home/pengziyang/MultimodalReasoningCompetition/Task1/datasets/VG_RS/VG-RS-images/{image_path}")
            input_height, input_width = smart_resize(
                image.height,
                image.width,
                28,
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
            )
            candidate_boxes = parse_json(annocandi['problem'])
            candidate_boxes_resized = []
            # reisze to raw image h w
            for _, box in enumerate(candidate_boxes):
                norm_box = resize2image(box, input_height, input_width, image.height, image.width)
                candidate_boxes_resized.append(norm_box)
                
            box_32b = [anno72b['result'][0][0],anno72b['result'][0][1],anno72b['result'][1][0],anno72b['result'][1][1]]
            best_matched = return_maximum_overlap(gt_box=box_32b, candidate_boxes=candidate_boxes_resized, min_iou=0.25)

            if best_matched is not None:
                if compute_iou(best_matched, box_32b) >= 0.3:
                    # convert abs coords to normalized 0~1
                    box_32b_norm = abs2propo(box_32b, image.height, image.width)
                    best_matched_norm = abs2propo(best_matched, image.height, image.width)
                    
                    # if any(num > 1.0 for num in box_32b_norm) or any(num > 1.0 for num in best_matched_norm):
                    #     print(f"image hw:{image.height} {image.width} | box_32b:{box_32b} | best_match:{best_matched}")
                    #     print(f"abnorm image key:{image_key} | text key:{text_key}")
                    # if box_32b_norm[0] == box_32b_norm[2] or box_32b_norm[1] == box_32b_norm[3] or \
                    #     best_matched_norm[0] == best_match   ed_norm[2] or best_matched_norm[1] == best_matched_norm[3]:
                    #     print(f"image hw:{image.height} {image.width} | box_32b:{box_32b} | best_match:{best_matched}")
                    #     print(f"abnorm image key:{image_key} | text key:{text_key}")
                        
                    boxes_list = [[box_32b_norm], [best_matched_norm]]
                    scores_list = [[0.8], [0.8]]
                    labels_list = [[1], [1]]
                    weights = [1, 2]
                    boxes, _, _ = weighted_boxes_fusion(boxes_list, scores_list, labels_list, 
                                weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
                    
                    if len(boxes) > 0:
                        boxes = boxes.tolist()[0]
                    else:
                        boxes = box_32b_norm
                    # de-normalized to raw image h w
                    boxes = propo2abs(boxes, image.height, image.width)
                    anno72b['result'][0] = boxes[:2]
                    anno72b['result'][1] = boxes[2:]
                    cnt_gdino_fusion += 1
                else:
                    anno72b['result'][0] = best_matched[:2]
                    anno72b['result'][1] = best_matched[2:]
                    cnt_gdino_replace += 1
                if False:
                    draw = ImageDraw.Draw(image)
                    coords_32b = [tuple(box_32b[:2]), tuple(box_32b[2:])]
                    coords_gdino = [tuple(best_matched[:2]), tuple(best_matched[2:])]
                    coords_fusion = [tuple(boxes[:2]), tuple(boxes[2:])]

                    draw.rectangle(coords_32b, outline="red", width=3)
                    draw.rectangle(coords_gdino, outline="green", width=3)
                    draw.rectangle(coords_fusion, outline="blue", width=3)
                    image.save(os.path.join(image_save_path, f"[{cnt}] " + image_path))
                cnt += 1
                
            else:
                # glm fusion
                for glm in anno_glm:
                    if glm['image_path'].split('\\')[1] == image_key and glm['question'].replace(".", "").strip() == text_key:
                        glm_box = [glm['result'][0][0], glm['result'][0][1],glm['result'][1][0],glm['result'][1][1]]
                        iou = compute_iou(box_32b, glm_box)
                        if iou >= 0.3:
                            box_32b_norm = abs2propo(box_32b, image.height, image.width)
                            best_matched_norm = abs2propo(glm_box, image.height, image.width)
                            boxes_list = [[box_32b_norm], [best_matched_norm]]
                            scores_list = [[0.8], [0.8]]
                            labels_list = [[1], [1]]
                            weights = [1, 2]
                            boxes, _, _ = weighted_boxes_fusion(boxes_list, scores_list, labels_list, 
                                        weights=weights, iou_thr=0.3, skip_box_thr=skip_box_thr)
                            
                            if len(boxes) > 0:
                                boxes = boxes.tolist()[0]
                            else:
                                boxes = box_32b_norm
                            boxes = propo2abs(boxes, image.height, image.width)
                            anno72b['result'][0] = boxes[:2]
                            anno72b['result'][1] = boxes[2:]
                            cnt_glm_fusion += 1
                        else:
                            if glm_replace_cnt <= 600:
                                anno72b['result'] = glm['result']
                                cnt_glm_replace += 1
                                glm_replace_cnt += 1
                            
                        glm_collect.append({"image_key":image_key, "text_key":text_key})
                        break

                # 32B correct -> append to vgrs hint list so we can add the correct box to candidate list
                # 32B worng -> append to vgrs hint list, maybe do not affect model's final choice
                box_32b_resized = resize2input(box_32b, input_height, input_width, image.height, image.width)
                new_list = candidate_boxes + [box_32b_resized]
                # print(candidate_boxes, new_list)
                annocandi['problem'] = annocandi['problem'].replace(str(candidate_boxes), str(new_list))
                # print(f"add new box into candidate box, image key:{image_key}, text_key:{text_key}")
                edit_collect.append({"image_key":image_key, "text_key":text_key})
            
            break

with open("/home/pengziyang/MultimodalReasoningCompetition/Task1/datasets/VG_RS/0805_72b_fusion_glm_iou_0.25_7.json", 'w', encoding='utf-8') as f:
    json.dump(anno_72b, f, indent=4, ensure_ascii=False)
    
with open("/home/pengziyang/MultimodalReasoningCompetition/Task1/datasets/VG_RS/glm_collect.json", 'w', encoding='utf-8') as f:
    json.dump(glm_collect, f, indent=4, ensure_ascii=False)

# with open("/home/pengziyang/MultimodalReasoningCompetition/Task1/datasets/VG_RS/0730_vgrs_rex_for_infer_full.jsonl", "w", encoding='utf-8') as f:
#     for anno in anno_candi:
#         json.dump(anno, f, ensure_ascii=False)
#         f.write("\n")
        
with open("/home/pengziyang/MultimodalReasoningCompetition/Task1/datasets/VG_RS/edit_records.json", 'w', encoding='utf-8') as f:
    json.dump(edit_collect, f, indent=4, ensure_ascii=False)


print(cnt)
print(cnt_gdino_replace)
print(cnt_gdino_fusion)
# print(cnt_glm_replace)
print(cnt_glm_fusion)
# print(cnt_yolo)
