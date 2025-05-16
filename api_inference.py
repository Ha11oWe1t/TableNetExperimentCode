import os
import re
import json
import time
import requests
import traceback
import openai
import random
import http.client
from openai import OpenAI

from utils.loss import TextMetrics

os.environ["DASHSCOPE_API_KEY"] = ""

def upload_url(img_path, url="https://api.superbed.cn/upload"):
    is_success = False
    while not is_success:
        try:
            resp = requests.post(url, data={"token": "1c7d94be6fd849eda88da6dc5c4adca4"}, files={"file": open(img_path, "rb")})
            is_success = True
            print(resp)
            url = resp.json()["url"]
            return url
        except Exception as e:
            # 如果是filenotfounderror
            if "No such file or directory" in str(e):
                img_path = os.path.join("/paddle/datasets/pubtabnet/test", os.path.basename(img_path))
            traceback.print_exc()
            continue

train_data_path = "data_vl_train.json"
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
with open(train_data_path) as f:
    train_datas = json.load(f)
    random.shuffle(train_datas)
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

def get_shots(n):
    shotmessages = []
    ds = random.sample(train_datas, n)
    for data in ds:
        image_path = data["conversations"][0]["value"].split("<|vision_start|>")[1].split("<|vision_end|>")[0]
        url = upload_url(image_path)
        html = data["conversations"][1]["value"]
        shot_template = [
            {"role": "user", "content": [
                {"type": "text", "text": "读取图片中的表格内容，并将其转换为HTML格式的表格。HTML以<html><body><table>开头，以</table></body></html>结尾。只输出这个HTML，不要输出其他任何其他内容。"},
                {"type": "image_url",
                 "image_url": {"url": url}}
            ]},
            {"role": "assistant", "content": html}
        ]
        shotmessages.extend(shot_template)
    return shotmessages



def GPT_infer(url, max_try=99):
    prompt = """读取图片中的表格内容，并将其转换为HTML格式的表格。HTML以<html><body><table>开头，以</table></body></html>结尾，并且只在一行中输出。你的输出必须只包含以<html><body><table>开头，以</table></body></html>结尾的这个html，不要输出其他任何其他内容。"""
    payload = json.dumps({
  "model": "claude-3-7-sonnet-20250219",
  "messages": [
    {
      "role": "user",
      "content": f"{url} {prompt}"
    }
  ]
})
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer " + "",
    }
    for attempt in range(max_try):
        try:
            conn = http.client.HTTPSConnection("yunwu.ai")
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            response = res.read().decode("utf-8")
            response = json.loads(response)['choices'][0]['message']['content']
            pattern = r"<html>\s*<body>.*?<table>.*?</table>.*?</body>\s*</html>"

            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                response = match.group(0)
                return response
            else:
                print("No match found.")
                return None
        except Exception as e:
            traceback.print_exc()

def LLM_infer(url, max_try=99, shots=None):
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    for attempt in range(max_try):
        try:
            messages = [{"role": "user", "content": [
                    {"type": "text", "text": """读取图片中的表格内容，并将其转换为HTML格式的表格。HTML以<html><body><table>开头，以</table></body></html>结尾，并且只在一行中输出。你的输出必须只包含以<html><body><table>开头，以</table></body></html>结尾的这个html，不要输出其他任何其他内容。"""},
                    {"type": "image_url",
                     "image_url": {"url": url}}
                ]}]
            if shots:
                messages = shots + messages
            completion = client.chat.completions.create(
                #model="qwen-vl-max-2025-04-08",
                model="qwen2-vl-72b-instruct",
                messages=messages
            )
            response = completion.model_dump_json()
            response = json.loads(response)['choices'][0]['message']['content']
            pattern = r"<html>\s*<body>.*?<table>.*?</table>.*?</body>\s*</html>"

            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                response = match.group(0)
                return response
            else:
                print("No match found.")
                return None
        except Exception as e:
            traceback.print_exc()

        time.sleep(3)

count = 0
for i, data in enumerate(test_datas[:1000]):
    shots = get_shots(1)
    image_path = data["conversations"][0]["value"].split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    url = upload_url(image_path)
    html = data["conversations"][1]["value"]
    #print(html)
    response = LLM_infer(url, shots=shots)
    #response = GPT_infer(url)
    if response is None:
        continue
    print(f"response: {response}")
    sim = met.calculate_teds(html, response)
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
    if all_count and simple_count and complex_count and lined_count and lineless_count and colored_count and colorless_count:
        print(i, "all_sim:", all_sim / all_count, "simple_sim:", simple_sim / simple_count, "complex_sim:",
              complex_sim / complex_count, "lined_sim:", lined_sim / lined_count, "lineless_sim:",
              lineless_sim / lineless_count, "colored_sim:", colored_sim / colored_count, "colorless_sim:",
              colorless_sim / colorless_count)
