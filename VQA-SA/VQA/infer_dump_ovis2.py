from transformers import AutoModelForCausalLM
from typing import List, Dict
from Portable import osp
from tqdm import tqdm
from PIL import Image

import argparse
import torch
import json


parser = argparse.ArgumentParser()
# I
parser.add_argument("--vl-base",                type=str,           default=r"./sln/Ovis2-34B")
parser.add_argument("--image-base",             type=str,           default=r"./sln/VQA_SA")
parser.add_argument("--question-json",          type=str,           default=r"./sln/VQA_SA/VQA-SA-question-ENG.json")
# O
parser.add_argument("--dump-dir",               type=str,           default=r"./output/vqasa/ovis2_splits")
# C
parser.add_argument("--device",                 type=str,           default="auto")
parser.add_argument("--max-output-length",      type=int,           default=1024)

args = parser.parse_args()

## 语言服务器辅助变量
vl_base:str                         = args.vl_base
image_base:str                      = args.image_base
question_json:str                   = args.question_json
dump_dir:str                        = args.dump_dir
device:str                          = args.device
max_output_length:int               = args.max_output_length

MAX_PARTITION = 9

osp.ensure_dirs(dump_dir)


## 加载模型和配置
model = AutoModelForCausalLM.from_pretrained(vl_base,
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True,
                                             empty_init=False,
                                             device_map=device)
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

gen_kwargs = dict(
    max_new_tokens=max_output_length,
    do_sample=False,
    top_p=None,
    top_k=None,
    temperature=None,
    repetition_penalty=None,
    eos_token_id=model.generation_config.eos_token_id,
    pad_token_id=text_tokenizer.pad_token_id,
    use_cache=True
)

## 读入问题 构造多轮问答
with open(question_json, "r", encoding="utf8") as reader:
    data = json.load(reader)

mulq_dict: Dict[str, list] = dict()
key_seq: List[str] = list()
for question in data:
    image_path = question["image_path"]
    if image_path not in mulq_dict: 
        mulq_dict[image_path] = list()
        key_seq.append(image_path)
    mulq_dict[image_path].append(question["question"])

## token 预处理函数
def prepare(
        query: str, 
        images: List[Image.Image],
    ):
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=MAX_PARTITION)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]
    return input_ids, pixel_values, attention_mask

## 官方 COT prompt
# COT_SUFFIX = "Provide a step-by-step solution to the problem, and conclude with 'the answer is' followed by the final solution."
COT_SUFFIX = "请使用中文进行回答，提供解决问题的详细步骤，并在回答的末尾用‘答案是...’给出最终答案。"

## 推理流程
with torch.inference_mode():

    for image_suf in tqdm(key_seq):
        dump_path = osp.j(dump_dir, osp.refmt(image_suf.replace("images\\", ""), ".json"))
        if osp.exists(dump_path): continue
        image_path = osp.j(image_base, image_suf.replace("\\", "/"))
        images = [Image.open(image_path)]

        history = "这是一些当前图片问答的历史记录：\n\n"
        dump_contents = list()
        for question in mulq_dict[image_suf]:
            query = f"{history}\n<image>\n{question}\n{COT_SUFFIX}"
            input_ids, pixel_values, attention_mask = prepare(query, images)
            output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
            history = history + question + "\n\nOutput:\n" + output + "\n\n"
            dump_contents.append({
                "image_path": image_suf,
                "question": question,
                "result": output
            })
        
        with open(dump_path, "w", encoding="utf8") as writer:
            json.dump(dump_contents, writer, indent=4, ensure_ascii=False)
        
