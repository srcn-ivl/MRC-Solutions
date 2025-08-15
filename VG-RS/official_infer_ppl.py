import argparse
import random
from pathlib import Path

from tqdm import tqdm

import numpy as np
import os, cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import ast
import json


def get_args_parser():
    parser = argparse.ArgumentParser('Visual grounding', add_help=False)

    # dataset parameters
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--model_dir', default=r'/data3/HuggingFace/Qwen2.5-VL-32B-Instruct/') # path/to/your/local/model
    parser.add_argument('--json_path', default='/data2/pengziyang/data/VG_RS/VG-RS-question.json')#'./VG-RS-question.json'
    parser.add_argument('--json_save_path',
                        default='/data2/tangyin/models/0804_32B_resume.json')
    return parser

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def read_json_and_extract_fields(file_path='VG-RS-question.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def main(args):

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="auto"
    )
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     args.model_dir, torch_dtype="auto", device_map="auto"
    # )

    # fix the seed for reproducibility
    seed = args.seed
    print('seed')
    print(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    img_path = r"/data2/pengziyang/data/VG_RS/images/"
    data_infer = read_json_and_extract_fields(args.json_path)
    data_infer = data_infer[:]
    batch_size = args.batch_size
    # default processer
    # min_pixels = 256 * 28 * 28
    # max_pixels = 3840*3840
    processor = AutoProcessor.from_pretrained(args.model_dir)
    # processor = AutoProcessor.from_pretrained(args.model_dir)
    content_list = []
    temp_result = []
    
    with open("/data2/tangyin/models/0804_72B_previous_combined.json", 'r', encoding='utf-8') as f:
        exist_record = json.load(f)
    cur = {}    
    for it in exist_record:
        if cur.get(it['image_path'].split('\\')[-1], None) is not None:
            cur[it['image_path'].split('\\')[-1]].append(it['question'])
        else:
            cur[it['image_path'].split('\\')[-1]] = [it['question']]
            
    for i in tqdm(range(len(data_infer)//batch_size)):
        # if i/(len(data_infer)//batch_size) <= 0.72:
        #     continue
        if i % 10 ==0:
            print(i/(len(data_infer)//batch_size))
        messages_list = []
        text_query_list = []
        image_name_list = []
        flag = 0
        for i_batch in range(batch_size):

            image_name = data_infer[i * batch_size + i_batch].get('image_path').split('images\\')[1]
            image_name_list.append(image_name)
            text_query = data_infer[i * batch_size + i_batch].get('question')
            text_query_list.append(text_query)
            image_path = os.path.join(img_path, str(image_name.lower()))
            
            get_image_key = cur.get(image_name, None)
            if get_image_key is not None and text_query in get_image_key:
                print("find history record, skip it .........")
                flag = 1
                break
            # print(image_path)
            # ʹ��system prompt
            # system_prompt = "You are a helpful assistant"
            messages = [
                # {
                #     "role": "system",
                #     "content": system_prompt
                # },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text",
                         "text": "Please provide the bounding box coordinate of the region this sentence describes: {} and output it in JSON format".format(text_query)
                         },

                    ],
                }
            ]
            messages_list.append(messages)
            
        if flag:
            continue
        # Preparation for inference
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            padding_side='left',
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for i_batch in range(batch_size):
            bounding_boxes = parse_json(output_text[i_batch])
            try:
                json_output = ast.literal_eval(bounding_boxes)
            except Exception as e:
                end_idx = bounding_boxes.rfind('"}') + len('"}')
                truncated_text = bounding_boxes[:end_idx] + "]"
                try:
                    json_output = ast.literal_eval(truncated_text)
                except Exception as e:
                    # print(i)
                    print(['[Unable to detect samples ]', *[f'{i}']])
                    print(image_path)
                    print(output_text[i_batch])
                    continue
            # Iterate over the bounding boxes
            for j_index, bounding_box in enumerate(json_output):
                if j_index >= 1:
                    continue
                try :
                    len(bounding_box["bbox_2d"]) != 4
                except KeyError:
                    continue
                except TypeError:
                    continue
                try:
                    abs_y1 = bounding_box["bbox_2d"][1]
                    abs_x1 = bounding_box["bbox_2d"][0]
                    abs_y2 = bounding_box["bbox_2d"][3]
                    abs_x2 = bounding_box["bbox_2d"][2]
                except IndexError:
                    continue
                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1
                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1
                content = {
                    "image_path": 'images\\' + image_name_list[i_batch],
                    'question': text_query_list[i_batch],
                    "result": [[abs_x1, abs_y1], [abs_x2, abs_y2]]}
                # print(content)
                content_list.append(content)
        
        if len(temp_result) > 0:
            if not (temp_result[-1]['image_path'] == content_list[-1]['image_path'] and temp_result[-1]['question'] == content_list[-1]['question']):
                temp_result.append(content_list[-1])
        elif len(content_list) > 0:
            temp_result.append(content_list[-1])
        else:
            continue
        with open(args.json_save_path, 'w', encoding='utf-8') as f:
            json.dump(temp_result, f, indent=4, ensure_ascii=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Infer result', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
