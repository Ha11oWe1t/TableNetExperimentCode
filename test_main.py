from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel, LoraConfig, TaskType
from utils.loss import TextMetrics
import random
import json

prompt = "你是一个HTML助手，目标是读取用户输入的表格图片，转换成HTML序列"
local_model_path = "Qwen/Qwen2-VL-2B-Instruct"
finetuned_model_path = "PATH TO TRAINED MODEL"

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    finetuned_model_path, torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(local_model_path)

test_data_path = "data_vl_test.json"
simple_path = "simple_data_vl.json"
complex_path = "complex_data_vl.json"
lined_path = "lined_data_vl.json"
lineless_path = "lineless_data_vl.json"
colored_path = "colored_data_vl.json"
colorless_path = "colorless_data_vl.json"
met = TextMetrics()
all_sim = 0
simple_sim = 0
complex_sim = 0
lined_sim = 0
lineless_sim = 0
colored_sim = 0
colorless_sim = 0
all_count = 0
simple_count = 0
complex_count = 0
lined_count = 0
lineless_count = 0
colored_count = 0
colorless_count = 0
##
simple_test_data = []
complex_test_data = []
lined_test_data = []
lineless_test_data = []
colored_test_data = []
colorless_test_data = []
#path = test_data_path
with open(test_data_path) as f:
    test_datas = json.load(f)
    random.shuffle(test_datas)
with open(simple_path, "r") as f:
    simple_datas = json.load(f)
    for data in simple_datas:
        if data in test_datas:
            simple_test_data.append(data)
with open(complex_path, "r") as f:
    complex_datas = json.load(f)
    for data in complex_datas:
        if data in test_datas:
            complex_test_data.append(data)
with open(lined_path, "r") as f:
    lined_datas = json.load(f)
    for data in lined_datas:
        if data in test_datas:
            lined_test_data.append(data)
with open(lineless_path, "r") as f:
    lineless_datas = json.load(f)
    for data in lineless_datas:
        if data in test_datas:
            lineless_test_data.append(data)
with open(colored_path, "r") as f:
    colored_datas = json.load(f)
    for data in colored_datas:
        if data in test_datas:
            colored_test_data.append(data)
with open(colorless_path, "r") as f:
    colorless_datas = json.load(f)
    for data in colorless_datas:
        if data in test_datas:
            colorless_test_data.append(data)
for i, data in enumerate(test_datas):
    test_image_path = data["conversations"][0]["value"].split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    html = data["conversations"][1]["value"]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": test_image_path,
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": f"{prompt}"},
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
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    pred = output_text[0]
    print(html)
    print(pred)
    sim = met.calculate_teds(pred, html)
    all_sim += sim
    all_count += 1
    if data in simple_test_data:
        simple_sim += sim
        simple_count += 1
    if data in complex_test_data:
        complex_sim += sim
        complex_count += 1
    if data in lined_test_data:
        lined_sim += sim
        lined_count += 1
    if data in lineless_test_data:
        lineless_sim += sim
        lineless_count += 1
    if data in colored_test_data:
        colored_sim += sim
        colored_count += 1
    if data in colorless_test_data:
        colorless_sim += sim
        colorless_count += 1
    print(all_count, simple_count, complex_count, lined_count, lineless_count, colored_count, colorless_count)
    if all_count and simple_count and complex_count and lined_count and lineless_count and colored_count and colorless_count:
        print(i, "all_sim:", all_sim / all_count, "simple_sim:", simple_sim / simple_count, "complex_sim:", complex_sim / complex_count, "lined_sim:", lined_sim / lined_count, "lineless_sim:", lineless_sim / lineless_count, "colored_sim:", colored_sim / colored_count, "colorless_sim:", colorless_sim / colorless_count)
