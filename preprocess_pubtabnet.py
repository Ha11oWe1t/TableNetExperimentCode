import json
import os
import random


def rebuild_html_from_ppstructure_label(label_info):
    from html import escape

    html_code = label_info["html"]["structure"]["tokens"].copy()
    to_insert = [
        i for i, tag in enumerate(html_code) if tag in ("<td>", ">")
    ]
    for i, cell in zip(to_insert[::-1], label_info["html"]["cells"][::-1]):
        if cell["tokens"]:
            cell = [
                escape(token) if len(token) == 1 else token
                for token in cell["tokens"]
            ]
            cell = "".join(cell)
            html_code.insert(i + 1, cell)
    html_code = "".join(html_code)
    html_code = "<html><body><table>{}</table></body></html>".format(
        html_code)
    return html_code
### Code below is preprocessing code for pubtabnet to train qwen-vl


data_dir = "/home/zrl/datasets/pubtabnet/train"
anno_dir = "/home/zrl/datasets/pubtabnet/PubTabNet_2.0.0.jsonl"

# 处理智能体生成数据
conversations = []
datas = os.listdir(data_dir)
with open(anno_dir, "r", encoding="utf-8") as f:
    lines = f.readlines()
    lines = [l for l in lines if json.loads(l)["split"]=="train"]
    print(len(lines))
random.shuffle(lines)
len_train = int(len(lines) * 0.8)
train_lines = lines[:len_train]
val_lines = lines[len_train:]
print(len(train_lines))
print(len(val_lines))

for i, line in enumerate(train_lines):
    line = json.loads(line.strip())
    img_path = os.path.join(data_dir, line["filename"])
    conv = {
    "id": f"identity_{i+1}",
    "conversations": [
        {
            "from": "user",
            "value": f"COCO Yes: <|vision_start|>{img_path}<|vision_end|>"
        },
        {
            "from": "assistant",
            "value": rebuild_html_from_ppstructure_label(line)
        }
    ]
}
    conversations.append(conv)


# with open('data_vl.json', 'w', encoding='utf-8') as f:
#     json.dump(conversations, f, ensure_ascii=False, indent=2)
with open("pub_train.json", "w", encoding="utf-8") as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)

conversations = []
for i, line in enumerate(val_lines):
    line = json.loads(line.strip())
    img_path = os.path.join(data_dir, line["filename"])
    conv = {
    "id": f"identity_{i+1}",
    "conversations": [
        {
            "from": "user",
            "value": f"COCO Yes: <|vision_start|>{img_path}<|vision_end|>"
        },
        {
            "from": "assistant",
            "value": rebuild_html_from_ppstructure_label(line)
        }
    ]
}
    conversations.append(conv)


# with open('data_vl.json', 'w', encoding='utf-8') as f:
#     json.dump(conversations, f, ensure_ascii=False, indent=2)
with open("pub_val.json", "w", encoding="utf-8") as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)

###### Code below is for filtering conversations with img_path containing 'crawl', which is real-world data

import json

def filter_conversations_by_img_path(input_json_path, output_json_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 用于存储包含 'crawl' 的条目
    filtered_data = []

    for entry in data:
        try:
            convs = entry.get("conversations", [])
            for conv in convs:
                if conv.get("from") == "user":
                    user_value = conv.get("value", "")
                    # 提取 <|vision_start|> 与 <|vision_end|> 之间的 img_path
                    start = user_value.find("<|vision_start|>") + len("<|vision_start|>")
                    end = user_value.find("<|vision_end|>")
                    img_path = user_value[start:end].strip()

                    if "crawl" in img_path:
                        filtered_data.append(entry)
                        break  # 已经命中就不检查剩下的对话了
        except Exception as e:
            print(f"Error parsing entry: {e}")
            continue
    print(len(filtered_data))

    # 写入新文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"Filtered {len(filtered_data)} entries written to: {output_json_path}")

filter_conversations_by_img_path("data_vl_test.json", "crawl_data_vl_test.json")


##### Code below is for sampling a subset of the original dataset, keep the training dataset size the same as training on our TableNet

import json
import random

# 设置随机种子以保证可复现（可选）
random.seed(42)

# 文件路径
train_path = "/home/zrl/datasets/TableNet/pub_train.json"
val_path = "/home/zrl/datasets/TableNet/pub_val.json"

# 新文件路径
train_new_path = train_path.replace(".json", "_new.json")
val_new_path = val_path.replace(".json", "_new.json")

# 加载原始数据
with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(val_path, "r", encoding="utf-8") as f:
    val_data = json.load(f)

# 随机采样
train_sample = random.sample(train_data, 48000)
val_sample = random.sample(val_data, 5000)

# 写入新文件
with open(train_new_path, "w", encoding="utf-8") as f:
    json.dump(train_sample, f, ensure_ascii=False, indent=2)

with open(val_new_path, "w", encoding="utf-8") as f:
    json.dump(val_sample, f, ensure_ascii=False, indent=2)
