from transformers import AutoProcessor, AutoTokenizer, PreTrainedModel
from typing import List, Callable, Union, Type, Tuple, Dict
from peft import PeftModel, LoraConfig, TaskType
from . import message_constructor as mc
from Portable import osp
from PIL import Image

import importlib as il


VLM_BASE = r"../sln/SpaceOm"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_RES = [1280, 720]
DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEFAULT_MAX_NEW_TOKENS = 64

class LoRAs:

    def ap(path:str):
        with osp.WorkDir(osp.op_dir(__file__)): rst = osp.op_abs(path)
        return rst

vl_model: PreTrainedModel = None
vl_device: str = None
vl_processor: AutoProcessor = None
vl_tokenizer: AutoTokenizer = None
vl_proc_size: List[int] = DEFAULT_RES

def ensure_vl_base(
        base_path:str=None,
        vlm_class:Type[PreTrainedModel]=None,
        device:str=DEFAULT_DEVICE,
        torch_dtype:str="auto",
        kvss:Dict[str, dict]={}
    ) -> None:
    global vl_model, vl_device, vl_processor
    twd = None if base_path else osp.op_dir(__file__)
    if not base_path: base_path = VLM_BASE
    with osp.WorkDir(twd):
        base_path = osp.op_abs(base_path)
        vl_device = device
        if not vlm_class: 
            import json
            hf_config = osp.j(base_path, "config.json")
            with open(hf_config, "r", encoding="utf8") as reader:
                config:dict = json.load(reader)
            architecture = config.get("architectures")
            error_state = "vlm load failed, try indicate vlm_class manually."
            if not architecture or (isinstance(architecture, list) and len(architecture) > 1): 
                print("bad architecture:", error_state); return
            architecture = architecture[0]
            try: 
                m_t = il.import_module("transformers")
                vlm_class = getattr(m_t, architecture)
            except: print("bad import:", error_state); return
        kvs_model = kvss.get("model", {}); kvs_proc = kvss.get("processor", {})
        vl_model = vlm_class.from_pretrained(
            base_path, 
            torch_dtype=torch_dtype, 
            device_map=vl_device,
            **kvs_model
        )
        vl_processor = AutoProcessor.from_pretrained(
            base_path, 
            device_map=vl_device,
            **kvs_proc
        )
    print(f"vlm based on {vlm_class} is loaded")

def load_tokenizer(base_path:str=None) -> None:
    global vl_tokenizer, vl_device
    assert vl_device, "load vlm base first!"
    twd = None if base_path else osp.op_dir(__file__)
    if not base_path: base_path = VLM_BASE
    with osp.WorkDir(twd):
        vl_tokenizer = AutoTokenizer.from_pretrained(
            base_path,
            device_map=vl_device,
            use_fast=False,
            trust_remote_code=True
        )
    print(f"tokenizer loaded")

def attach_lora(
        lora_path:str,
        inference_mode:bool=True,
        over_configs:dict=None,
        merge_lora:bool=True
    ) -> None: 
    if not lora_path: return
    global vl_model, vl_device
    assert vl_model and vl_device, "load vlm base first!"
    if over_configs:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=over_configs.get("target_modules", DEFAULT_LORA_TARGETS),
            inference_mode=inference_mode,
            r=over_configs.get("r", 64),
            lora_alpha=over_configs.get("lora_alpha", 32),
            lora_dropout=over_configs.get("lora_dropout", 0.1),
            bias=over_configs.get("bias", False)
        )
    vl_model = PeftModel.from_pretrained(
        vl_model, 
        model_id=lora_path, 
        config=lora_config if over_configs else None, 
        torch_device=vl_device
    )
    if merge_lora: vl_model = vl_model.merge_and_unload()
    print(f"lora {osp.op_pbase(lora_path, 1)} attached")

def reset_process_size(size:List[int]=DEFAULT_RES) -> None:
    global vl_proc_size
    vl_proc_size = size

def qwen_vl_infer(
        messages:mc.Message, 
        max_new_tokens:int=DEFAULT_MAX_NEW_TOKENS
    ) -> str:
    import qwen_vl_utils as qvu
    import transformers
    global vl_model, vl_processor
    text = vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = qvu.process_vision_info(messages)
    inputs = vl_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs: transformers.feature_extraction_utils.BatchFeature
    inputs = inputs.to(vl_model.device)
    generated_ids = vl_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return "".join(output_text)

def construct_message(
        question:str,
        question_temp:Callable=lambda _: _,
        image_path:Union[str, Image.Image]=None,
        system_hint:str=None,
    ) -> mc.Message:
    global vl_proc_size
    user_utter = question_temp(question)
    message = mc.Message()
    if system_hint: message.add_role_block("system", mc.Content().add_text(system_hint))
    req_content = mc.Content()
    if image_path: req_content.add_image(image_path, vl_proc_size)
    req_content.add_text(user_utter)
    message.add_role_block("user", req_content)
    return message

def iou(box1:list, box2:list) -> float:
    xs1, ys1, xe1, ye1 = box1
    xs2, ys2, xe2, ye2 = box2
    a1 = (xe1 - xs1) * (ye1 - ys1)
    a2 = (xe2 - xs2) * (ye2 - ys2)
    xsi = max(xs1, xs2)
    ysi = max(ys1, ys2)
    xei = min(xe1, xe2)
    yei = min(ye1, ye2)
    ai = (max(xei - xsi, 0)) * (max(yei - ysi, 0))
    return ai / (a1 + a2 - ai)

def train_sample_process(
        example, 
        loader_func:Callable[..., Tuple[mc.Message, str]]=lambda _:_,
        MAX_LENGTH:int=8192
    ):
    """
    将数据集进行预处理
    """
    import qwen_vl_utils as qvu
    import torch
    global vl_processor, vl_tokenizer
    
    messages, answer = loader_func(example)
    text = vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # 获取文本
    image_inputs, video_inputs = qvu.process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = vl_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {key: (value.tolist() if key not in ["pixel_values", "image_grid_thw"] else value) for key, value in inputs.items() } # tensor -> list, 为了方便拼接
    response = vl_tokenizer(f"{answer}", add_special_tokens=False)
    input_ids = (inputs["input_ids"][0] + response["input_ids"] + [vl_tokenizer.pad_token_id])
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
    labels = ([-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [vl_tokenizer.pad_token_id])

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids).to(vl_processor.device)
    attention_mask = torch.tensor(attention_mask).to(vl_processor.device)
    labels = torch.tensor(labels).to(vl_processor.device)
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels,
        "pixel_values": inputs['pixel_values'], 
        "image_grid_thw": inputs['image_grid_thw'].squeeze(0)
    }

