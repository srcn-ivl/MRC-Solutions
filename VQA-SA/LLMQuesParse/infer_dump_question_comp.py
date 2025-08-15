from Portable from Portable import osp
from tqdm import tqdm

print(osp.cmd("echo a"))
exit()

import utils.message_constructor as mc
import utils.vlm_utils as vu
import utils.prompts as sp
import torch
import json
import re


llm_base = r"./sln/Qwen2.5-14B-Instruct"
device = "cuda:0"

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


json_path = r"./sln/VQA_SA/VQA-SA-question.json"
imq_dict = dict()
with open(json_path, "r", encoding="utf8") as reader:
    data = json.load(reader)
    for e in data: 
        if e["image_path"] not in imq_dict: imq_dict[e["image_path"]] = list()
        imq_dict[e["image_path"]].append(e["question"])

def parse_out(output, questions):
    out_seq = list()
    for i in range(len(questions)):
        cpattern = rf"{i + 1}、(.+)"
        result = re.findall(cpattern, output)
        ## 如果有序号检测不到 忽略对这个问题的修改 返回原问题集
        if not result: print(f"not found\n{output}"); return questions
        out_seq.append(result[0])
    return out_seq

output_dir = r"./output/ques_comp_patch"
osp.ensure_dirs(output_dir)

for image_path, questions in tqdm(imq_dict.items()):
    dump_path = osp.j(output_dir, osp.refmt(image_path.replace("images\\", ""), ".json"))
    if osp.exists(dump_path): continue
    message = mc.Message().add_role_block("system", sp.PREP_SYSTEM)\
                        .add_role_block("user", input_construct(questions))
    output = qwen25_infer(message, 1024)
    output = parse_out(output, questions)
    ## 如果问题的集合相同 视作没有修改 可以规避 LLM 输出顺序问题
    if set(questions) == set(output): continue
    dump_contents = [list(et) for et in zip(questions, output)]
    dump_contents = [el[0] if el[0] == el[1] else {"src": el[0], "rct": el[1]} for el in dump_contents]
    with open(dump_path, "w", encoding="utf8") as writer:
        json.dump(dump_contents, writer, indent=4, ensure_ascii=False)

