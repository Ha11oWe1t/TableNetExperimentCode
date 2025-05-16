import json
import os


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

generation_dir = "PATH TO GENERATION DATA/"
opensource_dir = "PATH TO OPENDATA/"
crawl_dir = "PATH TO CRAWLED DATA/"

# 处理智能体生成数据
conversations = []
simple_table_conversations = []
complex_table_conversations = []
colored_table_conversations = []
colorless_table_conversations = []
lined_table_conversations = []
lineless_table_conversations = []
for subdir in os.listdir(generation_dir):
    generation_subdir = generation_dir + subdir
    if not os.path.isdir(generation_subdir): continue
    generation_img_dir = generation_subdir + "/img/"
    generation_gt_path = generation_subdir + "/gt.txt"

    with open(generation_gt_path, "r", encoding="utf-8") as f:
        generation_gt = f.readlines()
    for i, line in enumerate(generation_gt):
        line = json.loads(line.strip())
        filename = line['filename'][4:]
        img_path = generation_img_dir + filename
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
        if line["is_simple"]:
            simple_table_conversations.append(conv)
        else:
            complex_table_conversations.append(conv)
        if line["is_colored"]:
            colored_table_conversations.append(conv)
        else:
            colorless_table_conversations.append(conv)
        if line["is_lined"]:
            lined_table_conversations.append(conv)
        else:
            lineless_table_conversations.append(conv)

#处理爬虫数据
crawl_gt_path = crawl_dir + "manual.jsonl"
crawl_datas = []
start = len(conversations)
with open(crawl_gt_path, "r", encoding="utf-8") as f:
    for line in f:
        j = json.loads(line.strip())
        crawl_datas.append(j)
for i, line in enumerate(crawl_datas):
    img_path = crawl_dir + line['filename']
    conv = {
        "id": f"identity_{i+1+start}",
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
    if line["is_simple"]:
        simple_table_conversations.append(conv)
    else:
        complex_table_conversations.append(conv)
    if line["is_colored"]:
        colored_table_conversations.append(conv)
    else:
        colorless_table_conversations.append(conv)
    if line["is_lined"]:
        lined_table_conversations.append(conv)
    else:
        lineless_table_conversations.append(conv)


opensource_datas = []
opensource_gt_path = opensource_dir + "TABMWP/gt.txt"
print(opensource_gt_path)
start = len(conversations)
with open(opensource_gt_path, "r", encoding="utf-8") as f:
    opensource_gt = f.readlines()
    for i, line in enumerate(opensource_gt):
        line = json.loads(line.strip())
        filename = line['filename']
        img_path = opensource_dir + filename
        conv = {
            "id": f"identity_{i+1+start}",
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
        if line["is_simple"]:
            simple_table_conversations.append(conv)
        else:
            complex_table_conversations.append(conv)
        if line["is_colored"]:
            colored_table_conversations.append(conv)
        else:
            colorless_table_conversations.append(conv)
        if line["is_lined"]:
            lined_table_conversations.append(conv)
        else:
            lineless_table_conversations.append(conv)

# with open('data_vl.json', 'w', encoding='utf-8') as f:
#     json.dump(conversations, f, ensure_ascii=False, indent=2)
with open('simple_data_vl.json', 'w', encoding='utf-8') as f:
    json.dump(simple_table_conversations, f, ensure_ascii=False, indent=2)
with open('complex_data_vl.json', 'w', encoding='utf-8') as f:
    json.dump(complex_table_conversations, f, ensure_ascii=False, indent=2)
with open('colored_data_vl.json', 'w', encoding='utf-8') as f:
    json.dump(colored_table_conversations, f, ensure_ascii=False, indent=2)
with open('colorless_data_vl.json', 'w', encoding='utf-8') as f:
    json.dump(colorless_table_conversations, f, ensure_ascii=False, indent=2)
with open('lined_data_vl.json', 'w', encoding='utf-8') as f:
    json.dump(lined_table_conversations, f, ensure_ascii=False, indent=2)
with open('lineless_data_vl.json', 'w', encoding='utf-8') as f:
    json.dump(lineless_table_conversations, f, ensure_ascii=False, indent=2)