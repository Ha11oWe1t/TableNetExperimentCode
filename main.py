import os
import argparse
import torch
from datasets import Dataset
#from datasets import disable_caching
from modelscope import snapshot_download, AutoTokenizer
from qwen_vl_utils import process_vision_info
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model, PeftModel, LoraConfig
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    #Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import json
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator, notebook_launcher
#from swanlab.integration.transformers import SwabLabCallback


os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
#disable_caching()
prompt = "你是一个HTML助手，目标是读取用户输入的表格图片，转换成HTML序列"

def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 获取图像路径
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # {'input_ids': tensor([[]]), 'attention_mask': , 'pixel_values': , 'image_grid_thw': }
    inputs = {key: value.tolist() for key, value in inputs.items()}  # tensor -> list,为了方便拼接
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)
    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values']).to(torch.bfloat16)
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  # 由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


# 使用Transformers加载模型权重
model_name = "Qwen/Qwen2-VL-2B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
#model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 处理数据集：读取json文件
# 拆分成训练集和测试集，保存为data_vl_train.json和data_vl_test.json
# json_path = "data_vl.json"
# test_ratio = 0.15
# val_ratio = 0.15
# with open(json_path, 'r') as f:
#     data = json.load(f)
#     data_len = len(data)
#     test_len = int(data_len * (test_ratio + val_ratio))
#     val_len = int(data_len * val_ratio)
#     test_data = random.sample(data, test_len)
#     train_data = [item for item in data if item not in test_data]
#     val_data = random.sample(test_data, val_len)
#     test_data = [item for item in test_data if item not in val_data]
#
# with open("data_vl_train.json", "w") as f:
#     json.dump(train_data, f)
#
# with open("data_vl_test.json", "w") as f:
#     json.dump(test_data, f)
#
# with open("data_vl_val.json", "w") as f:
#     json.dump(val_data, f)

train_ds = Dataset.from_json("data_vl_train.json")
#train_ds = train_ds.select(range(30000))
train_dataset = train_ds.map(process_func)
eval_ds = Dataset.from_json("data_vl_val.json")
eval_dataset = eval_ds.map(process_func)
ftmethod = "full"

if ftmethod == "lora":
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=64,  # Lora 秩
        lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.05,  # Dropout 比例
        bias="none",
    )
elif ftmethod == "prompt":
    config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        #prompt_tuning_init_text="Please analyze the structure of the table and the content of its cells in the image below, and generate the corresponding HTML sequence for that table.",  # 设置hard_prompt的具体内容
        #num_virtual_tokens=len(tokenizer("Please analyze the structure of the table and the content of its cells in the image below, and generate the corresponding HTML sequence for that table.")["input_ids"]),
        num_virtual_tokens=4,
        tokenizer_name_or_path=model_name
    )



# 获取LoRA模型
#peft_model = get_peft_model(model, config)

# 配置训练参数
training_args = TrainingArguments(
    output_dir=f"./output/{model_name}-{ftmethod}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    eval_steps=200,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=False,
    report_to="none",
    save_strategy="steps",
    eval_strategy="steps",
    save_total_limit=5,
    ddp_find_unused_parameters=False,
)

# swanlab_callback = SwabLabCallback(
#     project="Qwen2-VL-tablenet",
#     experiment_name="2B-alldata",
#
# )

# 配置Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
# 开启模型训练
trainer.train(
    resume_from_checkpoint=True
)

# ===测试模式===
# 配置测试参数
# val_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     inference_mode=True,  # 训练模式
#     r=64,  # Lora 秩
#     lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
#     lora_dropout=0.05,  # Dropout 比例
#     bias="none",
# )
#
# # 获取测试模型
# val_peft_model = PeftModel.from_pretrained(model, model_id="./output/Qwen2-VL-2B/checkpoint-prompt-tuning-100", config=val_config)
#
# # 读取测试数据
# with open("data_vl_test.json", "r") as f:
#     test_dataset = json.load(f)
#
# test_image_list = []
# for item in test_dataset:
#     input_image_prompt = item["conversations"][0]["value"]
#     # 去掉前后的<|vision_start|>和<|vision_end|>
#     origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
#
#     messages = [{
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": origin_image_path
#             },
#             {
#                 "type": "text",
#                 "text": "COCO Yes:"
#             }
#         ]}]
#
#     response = predict(messages, val_peft_model)
#     messages.append({"role": "assistant", "content": f"{response}"})
#     print(messages[-1])