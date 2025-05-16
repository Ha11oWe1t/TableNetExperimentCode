from openai import OpenAI
import json
import re

def test_end2end(no_of_rows, no_of_cols, row_spans_matrix, col_span_matrix, is_simple, has_content, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    if has_content:
        content = "有单元格内容的"
    else:
        content = "没有单元格内容的"
    prompt = f"""请生成一个{no_of_rows}行{no_of_cols}列的{content}表格HTML, 序列中只需要使用<tr>,<td>,</tr>和</td>标签"""
    prompt += """请将生成的表格以json的格式返回给我：{"html": "表格HTML"}。"""
    if not is_simple:
        prompt += "表格中应该有单元格跨行跨列的情况，下面是两个矩阵，分别表示行跨行和列跨列的情况：\n" + \
            f"行跨行矩阵：{row_spans_matrix}\n" + \
            f"列跨列矩阵：{col_span_matrix}\n"
    completion = client.chat.completions.create(
        model="deepseek-r1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    response = json.loads(completion.model_dump_json())
    response = response["choices"][0]["message"]["content"]
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        json_str = match.group()
        data = json.loads(json_str)
        html = data.get("html")
        if html:
            return html
    return """<tr></tr>"""

def GPT_infer(no_of_rows, no_of_cols, row_spans_matrix, col_span_matrix, is_simple, has_content, api_key):
    prompt = f"""请生成一个{no_of_rows}行{no_of_cols}列的{content}表格HTML, 序列中只需要使用<tr>,<td>,</tr>和</td>标签"""
    prompt += """请将生成的表格以json的格式返回给我：{"html": "表格HTML"}。"""
    if not is_simple:
        prompt += "表格中应该有单元格跨行跨列的情况，下面是两个矩阵，分别表示行跨行和列跨列的情况：\n" + \
                  f"行跨行矩阵：{row_spans_matrix}\n" + \
                  f"列跨列矩阵：{col_span_matrix}\n"
    payload = json.dumps({
  "model": "gpt-4o-all",
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
        "Authorization": "Bearer " + "sk-7quUyaXtooGszCzmZ9lHTZqQKQvj0KKETBiq9rtpcjfwCVbW",
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