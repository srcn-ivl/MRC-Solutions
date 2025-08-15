from Portable import osp
from tqdm import tqdm

import utils.message_constructor as mc
import utils.vlm_utils as vu
import utils.prompts as sp
import json_repair as jr
import torch
import json


json_path = r"./sln/VQA_SA/VQA-SA-question.json"
llm_base = r"./sln/qwen2.5_lm_7B"
out_dir = r"./output/parsed"
device = "cuda:0"

osp.ensure_dirs(out_dir)

with open(json_path, "r", encoding="utf8") as reader:
    data = json.load(reader)
if_dict = dict()
key_seq = list()
for e in data:
    if e["image_path"] not in if_dict:
        if_dict[e["image_path"]] = list()
        key_seq.append(e["image_path"])
    if_dict[e["image_path"]].append(e["question"])

vu.ensure_vl_base(
    base_path=llm_base,
    device=device,
    torch_dtype=torch.bfloat16,
)

llm_model = vu.vl_model
llm_processor = vu.vl_processor

def qwen25_infer(
        messages:mc.Message,
        max_new_tokens:int=vu.DEFAULT_MAX_NEW_TOKENS
    ) -> str:
    text = llm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_processor(text=[text], return_tensors="pt")
    inputs = inputs.to(llm_model.device)
    generated_ids = llm_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = llm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return "".join(output_text)

def input_construct(ques_list:list) -> str:
    ques_list = [f"{(i + 1)}、{ques_list[i]}" for i in range(len(ques_list))]
    return "\n".join(ques_list)

errors = list()
for key in tqdm(key_seq):
    user_utter = input_construct(if_dict[key])
    message = mc.Message().add_role_block("system", sp.PARSER_SYSTEM)\
                .add_role_block("user", user_utter)
    output = qwen25_infer(message, 2048)
    output = output.replace("```json", "").replace("```", "")
    try:
        output = jr.repair_json(output)
        with open(osp.j(out_dir, osp.refmt(key, ".json").replace("images\\", "")), "w", encoding="utf8") as writer:
            json_o:list = json.loads(output)
            dump_contents = [{"question": q, "parse": json_o[i]} for i, q in enumerate(if_dict[key])]
            json.dump(dump_contents, writer, indent=4, ensure_ascii=False)
    except:
        errors.append(key)
        continue

print("\n" * 10, "errors:\n")
for err in errors: print(err)

