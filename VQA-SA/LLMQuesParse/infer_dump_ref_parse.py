from Portable import osp
from tqdm import tqdm

import utils.message_constructor as mc
import utils.vlm_utils as vu
import utils.prompts as sp
import torch
import json


llm_base = r"./sln/Qwen2.5-14B-Instruct"
device = "auto"

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

json_path = r"./output/result/direction_sort/ques_sort.json"
image_dict = dict()
with open(json_path, "r", encoding="utf8") as reader:
    data = json.load(reader)
    for el in data: 
        if el[0] not in image_dict: image_dict[el[0]] = list()
        image_dict[el[0]].append(el[1])

dump_dir = r"./output/ref_ext"
for image_name, questions in tqdm(image_dict.items()):
    dump_path = osp.j(dump_dir, osp.refmt(image_name, ".json"))
    if osp.exists(dump_path): continue
    dump_contents = list()
    for question in questions:
        user_utter = f"""问题：{question}"""
        message = mc.Message().add_role_block("system", sp.REF_SYSTEM)\
                            .add_role_block("user", user_utter)
        output = qwen25_infer(message, 256)
        dump_contents.append({
            "question": question,
            "parsed": output
        })
    with open(dump_path, "w", encoding="utf8") as dumper:
        json.dump(dump_contents, dumper, indent=4, ensure_ascii=False)

