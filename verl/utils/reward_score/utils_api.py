import os
import json
import re
from collections import defaultdict
from multiprocessing import Process
import hmac
import hashlib
from mimetypes import guess_type
import datetime
import uuid
import sys

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


import json
import os
import copy
import base64
from PIL import Image
import io
import requests
from tqdm import tqdm
import random
import concurrent.futures
from typing import Dict, Any, Tuple
from loguru import logger
import threading
import json
from PIL import Image
import io
import base64
import time
from .utils import print_hl, print_error

logger.add("gemini_similar.log", level="DEBUG")

file_lock = threading.Lock()

name2key = {
    "api_doubao_doubao-1-5-thinking-vision-pro-250428": {
        "model_marker": "api_doubao_doubao-1-5-thinking-vision-pro-250428",
        "api_key": "e1b1dbe4-e4ca-4c5a-83d7-2c92701d9e47",
    },
    "api_doubao_Doubao-Seed-1.6-thinking-250615": {
        "model_marker": "api_doubao_Doubao-Seed-1.6-thinking-250615",
        "api_key": "c99f635d-9b9e-41d7-8841-9dca490be825",
    },
    "api_google_gemini-2.5-pro-preview-05-06": {
        "model_marker": "api_google_gemini-2.5-pro-preview-05-06",
        "api_key": "c48d1677-57c0-4af1-a305-5dd2a99cbed9",
    },
    "api_openai_chatgpt-4o-latest": {
        "model_marker": "api_openai_chatgpt-4o-latest",
        "api_key": "aa57e5e3-92de-4d6c-8e60-8f57c6d4ec4a",
    },
    "api_google_gemini-2.5-flash-preview-05-20": {
        "model_marker": "api_google_gemini-2.5-flash-preview-05-20",
        "api_key": "d5b00969-d953-4043-9fc7-27f4af0040eb",
    },
    "api_openai_gpt-4.1": {
        "model_marker": "api_openai_gpt-4.1",
        "api_key": "4b810677-be38-4aee-aab3-5ae6ac36b7f7",
    },
    "api_anthropic_claude-3-7-sonnet-20250219": {
        "model_marker": "api_anthropic_claude-3-7-sonnet-20250219",
        "api_key": "9fa85b93-92d3-4106-a09e-27c5025c08db",
    },
    "api_yuewen_step-1o-vision-32k": {
        "model_marker": "api_yuewen_step-1o-vision-32k",
        "api_key": "634a5740-ad90-4078-b802-aa46653907c3",
    },
    "claude-3-5-sonnet@20240620": {
        "model_marker": "api_google_claude-3-5-sonnet@20240620",
        "api_key": "ae2273c6-21ff-4abc-a906-4e2a3ea54a3e"
    },
    "gpt-4o-2024-05-13": {
        "model_marker": "api_openai_gpt-4o-2024-05-13",
        "api_key": "c69d1207-1008-499a-a966-c4d372d2d832"
    },
    # "api_google_gemini-2.5-pro": {
    #     "model_marker": "api_google_gemini-2.5-pro",
    #     "app_id": "10Vo5UZ5_peterrao",
    #     "api_key": "3vQhoVPn2RBAqyCD",
    # },
    # "api_doubao_DouBao-Seed-1.6-Vision-250815": {
    #     "model_marker": "api_doubao_DouBao-Seed-1.6-Vision-250815",
    #     "app_id": "10Vo5UZ5_peterrao",
    #     "api_key": "3vQhoVPn2RBAqyCD",
    # },
    "api_azure_openai_gpt-5-chat-latest": {
        "model_marker": "api_azure_openai_gpt-5-chat-latest",
        "app_id": "10Vo5UZ5_peterrao",
        "api_key": "3vQhoVPn2RBAqyCD",
    },
    # "api_azure_openai_gpt-4o-mini": {
    #     "model_marker": "api_azure_openai_gpt-4o-mini",
    #     "app_id": "10Vo5UZ5_peterrao",
    #     "api_key": "3vQhoVPn2RBAqyCD",
    # },
    # 'azure-gpt-4o-mini': {
    #     "model_marker": "api_azure_openai_gpt-4o-mini",
    #     "app_id": "10Vo5UZ5_peterrao",
    #     "api_key": "3vQhoVPn2RBAqyCD",
    # },
    "api_openai_o4-mini": {
        "model_marker": "api_openai_o4-mini", 
        "app_id": "mKLHmg3Y_berlinni",
        "api_key": "XVWuzZxVY10QdEqk",
    },
    "api_openai_gpt-5-mini": {
        "model_marker": "api_openai_gpt-5-mini", 
        "app_id": "mKLHmg3Y_berlinni",
        "api_key": "XVWuzZxVY10QdEqk",
    },
    "api_openai_gpt-5-nano": {
        "model_marker": "api_openai_gpt-5-nano", 
        "app_id": "mKLHmg3Y_berlinni",
        "api_key": "XVWuzZxVY10QdEqk",
    },
    "api_azure_openai_gpt-4o-mini": {
        "model_marker": "api_azure_openai_gpt-4o-mini", 
        "app_id": "mKLHmg3Y_berlinni",
        "api_key": "XVWuzZxVY10QdEqk",
    },
    # 1105
    "api_azure_openai_gpt-5-nano": {
        "model_marker": "api_azure_openai_gpt-5-nano", 
        "app_id": "Py7UQ1Ss_peterrao",
        "api_key": "yA1n5zSbXa9cxMsd",
    },
    "api_azure_openai_gpt-5-nano-response": {
        "model_marker": "api_azure_openai_gpt-5-nano-response", 
        "app_id": "Py7UQ1Ss_peterrao",
        "api_key": "yA1n5zSbXa9cxMsd",
    },
    "api_azure_openai_gpt-5-mini": {
        "model_marker": "api_azure_openai_gpt-5-mini", 
        "app_id": "Py7UQ1Ss_peterrao",
        "api_key": "yA1n5zSbXa9cxMsd",
    },
    "api_azure_openai_gpt-5-mini-response": {
        "model_marker": "api_azure_openai_gpt-5-mini-response", 
        "app_id": "Py7UQ1Ss_peterrao",
        "api_key": "yA1n5zSbXa9cxMsd",
    },
    "api_azure_openai_gpt-4o-mini-batch": {
        "model_marker": "api_azure_openai_gpt-4o-mini-batch", 
        "app_id": "mKLHmg3Y_berlinni",
        "api_key": "XVWuzZxVY10QdEqk",
    },
    "api_doubao_DouBao-Seed-1.6-Vision-250815": {
        "model_marker": "api_doubao_DouBao-Seed-1.6-Vision-250815",
        "app_id": "mKLHmg3Y_berlinni",
        "api_key": "XVWuzZxVY10QdEqk",
    },
    "api_google_gemini-2.5-pro": {
        "model_marker": "api_google_gemini-2.5-pro",
        "app_id": "Py7UQ1Ss_peterrao",
        "api_key": "yA1n5zSbXa9cxMsd",
    },
    "api_google_gemini-2.5-flash": {
        "model_marker": "api_google_gemini-2.5-flash",
        "app_id": "Py7UQ1Ss_peterrao",
        "api_key": "yA1n5zSbXa9cxMsd",
    },

    # "api_openai_o4-mini": {
    #     "model_marker": "api_openai_o4-mini", 
    #     "app_id": "10Vo5UZ5_peterrao",
    #     "api_key": "3vQhoVPn2RBAqyCD",
    # },

    "api_azure_openai_gpt-5": {
        "model_marker": "api_azure_openai_gpt-5",
        "app_id": "10Vo5UZ5_peterrao",
        "api_key": "3vQhoVPn2RBAqyCD",
    },
    "t3v_sft0923": {
        "model_marker": "t3v_sft0923",
        "app_id": "10Vo5UZ5_peterrao",
        "api_key": "3vQhoVPn2RBAqyCD",
    }
}

NANO_MODEL_LIST = [
    # "api_openai_o4-mini",
    # "api_azure_openai_gpt-4o-mini",
    # "api_openai_gpt-5-mini",
    # "api_openai_gpt-5-nano",

    # "api_azure_openai_gpt-4o-mini-batch",

    "api_azure_openai_gpt-5-nano-response",
    "api_azure_openai_gpt-5-mini-response",
    "api_azure_openai_gpt-5-mini",
    "api_azure_openai_gpt-5-nano",
]


def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_simple_auth(source, SecretId, SecretKey):
    dateTime = datetime.datetime.now(datetime.timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
    auth = "hmac id=\"" + SecretId + "\", algorithm=\"hmac-sha1\", headers=\"date source\", signature=\""
    signStr = "date: " + dateTime + "\n" + "source: " + source
    sign = hmac.new(SecretKey.encode(), signStr.encode(), hashlib.sha1).digest()
    sign = base64.b64encode(sign).decode()
    sign = auth + sign + "\""
    return sign, dateTime

def get_header(app_id, api_key):
    source = 'xxxxxx'  # 签名水印值，可填写任意值
    sign, dateTime = get_simple_auth(source, app_id, api_key)
    headers = {
        'Apiversion': "v2.03",
        'Authorization': sign,
        'Date': dateTime,
        'Source': source
    }
    return headers


def _to_proxy_message_content(messages):
    """
    将 eval_mat 风格的输入转换为后端期望的:
    [{"role":"user","content":[{"type":"image_url","value":"data:..."}, {"type":"text","value":"..."}]}]
    同时抽取 system prompt 字符串并返回 (sys_message, messages)。
    """
    sys_message = ""
    out_messages = []

    for m in messages or []:
        role = m.get('role')
        content = m.get('content')

        # 抽取并合并 system 文本
        if role == 'system':
            if isinstance(content, str):
                sys_message += content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        txt = item.get('text') or item.get('value') or ''
                        sys_message += txt
            elif isinstance(content, dict):
                txt = content.get('text') or content.get('value') or ''
                sys_message += txt
            continue

        normalized = []

        def push_text(txt):
            if txt is None:
                return
            s = str(txt)
            if s != "":
                normalized.append({"type": "text", "value": s})

        def push_image(url):
            if not url:
                return
            try:
                if isinstance(url, str) and url.startswith('data:image/'):
                    normalized.append({"type": "image_url", "value": url})
                elif isinstance(url, str) and re.match(r'^https?://', url, re.I):
                    # 远程链接直接透传
                    normalized.append({"type": "image_url", "value": url})
                else:
                    # 认为是本地/相对路径，转 b64
                    b64 = encode_image_to_base64(url)
                    normalized.append({"type": "image_url", "value": f"data:image/jpeg;base64,{b64}"})
            except Exception as e:
                logger.error(f"Failed to convert image '{url}' to base64: {e}")

        # 规范化 content => 列表
        if isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    push_text(item)
                    continue
                if not isinstance(item, dict):
                    push_text(item)
                    continue
                t = item.get('type')
                if t == 'text' or (t is None and ('text' in item or 'value' in item)):
                    push_text(item.get('text') or item.get('value'))
                elif t == 'image_url':
                    url = None
                    img = item.get('image_url')
                    if isinstance(img, dict):
                        url = img.get('url') or img.get('value')
                    url = url or item.get('url') or item.get('value')
                    push_image(url)
                else:
                    push_text(item.get('text') or item.get('value'))
        else:
            if isinstance(content, str):
                push_text(content)
            elif isinstance(content, dict):
                t = content.get('type')
                if t == 'text':
                    push_text(content.get('text') or content.get('value'))
                elif t == 'image_url':
                    url = None
                    img = content.get('image_url')
                    if isinstance(img, dict):
                        url = img.get('url') or img.get('value')
                    url = url or content.get('url') or content.get('value')
                    push_image(url)
                else:
                    push_text(content.get('text') or content.get('value'))
            elif content is not None:
                push_text(content)

        out_messages.append({"role": role, "content": normalized})

    return sys_message, out_messages


def call_model_turn(messages, model_name, timeout=120, try_num=5):
    app_id = name2key[model_name]["app_id"]
    api_key = name2key[model_name]["api_key"]
    model_marker = name2key[model_name]["model_marker"]
    
    headers = dict(get_header(app_id, api_key))
    sys_prompt, messages = _to_proxy_message_content(messages)

    json_data = {
        "request_id": str(uuid.uuid4()),
        "model_marker": model_marker,
        "params": {
            "model": model_name,
        },
        "timeout": timeout,
        "messages": messages,
        "system": sys_prompt,
    }
    host = "http://trpc-gpt-eval.production.polaris:8080"
    url = host + '/api/v1/data_eval'

    try_num = try_num
    while try_num > 0:
        try:
            # import pdb; pdb.set_trace()
            res = requests.post(url=url, json=json_data, headers=headers)
            # import pdb; pdb.set_trace()
            # print(json.dumps(res.json(), ensure_ascii=False, indent=4))
            if res.status_code == 200 and res is not None:
                if len(res.json()['answer']) > 1:
                    # return(res.json()['answer'][0]['value'] + "\n\n" + res.json()['answer'][1]['value'])
                    return res.json()['answer'][1]['value']
                else:
                    return (res.json()['answer'][0]['value'])
                # return res.json()['answer'][0]['value'], res.json()['answer'][1]['value']
        except Exception as e:
            logger.error(f"Error in calling {model_name}: {e}. try_num: {try_num}")
            sys.stderr.flush()
            sys.stdout.flush()
            if 'none' in str(e).lower():
                break

            try_num -= 1
            time.sleep(5-try_num)
            continue
    logger.error(f"Fianlly failed in calling {model_name}")
    return None


def chat_gpt5_chat(messages):
    return call_model_turn(messages, "api_azure_openai_gpt-5-chat-latest")


def chat_gpt5(messages):
    return call_model_turn(messages, "api_azure_openai_gpt-5")


def chat_seedvl(messages):
    return call_model_turn(messages, "api_doubao_DouBao-Seed-1.6-Vision-250815")


def chat_gemini(messages):
    return call_model_turn(messages, "api_google_gemini-2.5-pro")


def chat_gemini_flash(messages):
    return call_model_turn(messages, "api_google_gemini-2.5-flash")


def chat_t3v(messages):
    return call_model_turn(messages, "t3v_sft0923")


def chat_4o_mini(messages, timeout=None, api_key=None, debug=False):
    for m in NANO_MODEL_LIST:
        if debug:
            print_hl(m)
            print(call_model_turn(messages, m, timeout=5, try_num=1))
        else:
            rst = call_model_turn(messages, m, 
                            timeout=(10 if debug else 10),
                            try_num=(1 if debug else 3))
            if rst:
                if m != NANO_MODEL_LIST[0]:
                    print_hl(f'applied {m}, expected {NANO_MODEL_LIST[0]}')
                return rst
    raise ValueError('No 4o-mini!!')

    
    # return call_model_turn(messages, 'azure-gpt-4o-mini')


if __name__ == '__main__':
    # 1) 生成一张临时测试图片，确保本地路径可被转为 base64
    # tmp_img_dir = "/mnt/private/agent_workspace/hunyuan-o3/.temp/tests/to_proxy"
    # os.makedirs(tmp_img_dir, exist_ok=True)
    tmp_img_path = "/mnt/private/agent_workspace/hunyuan-o3/.temp/datasets/gemini-workflow-geobench-0825/c8e1145d-5147-40c1-88a5-1efc09a2c696/panorama-c8e1145d-5147-40c1-88a5-1efc09a2c696.png"
    # if not os.path.exists(tmp_img_path):
    #     img = Image.new("RGB", (64, 64), (200, 160, 90))
    #     img.save(tmp_img_path, format="JPEG")

    # 2) 构造测试 messages（含 system、文本、路径图片、HTTPS 图片、assistant 工具调用字符串）
    test_messages = [
        # {
        #     "role": "system",
        #     "content": "You are a geolocation assistant. Please use tools to analyze the image."
        # },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": tmp_img_path}},
                # {"type": "text", "text": "the original resolution is 64x64"}
            ]
        },
        {
            "role": "assistant",
            "content": "<think>zoom candidate</think>\n<tool_call>\n{\"name\":\"image_zoom_in_tool\",\"arguments\":{\"bbox_2d\":[10,10,30,30]}}\n</tool_call>"
        },
        {
            "role": "user",
            "content": [
                # {"type": "image_url", "image_url": {"url": ""}}
                # {"type": "text", "text": "Please analyze where is the place."}
                {"type": "text", "text": "Who are you? Do you have web search access? Do you have crop zoom in tool access?"}
            ]
        },
    ]

    # 3) 转换并打印结果
    # sys_message, proxied = _to_proxy_message_content(test_messages)
    # print("=== sys_message ===")
    # print(sys_message)
    # print("=== proxied messages ===")
    # print(json.dumps(proxied, ensure_ascii=False, indent=2))
    # rst = chat_seedvl(test_messages)
    # print_hl(chat_seedvl.__name__ + " >")
    # print(rst)
    rst = chat_4o_mini(test_messages, debug=False)
    print_hl(chat_4o_mini.__name__ + " >")
    print(rst)
    # rst = chat_gpt5_chat(test_messages)
    # print_hl(chat_gpt5.__name__ + " >")
    # print(rst)
    rst = chat_gemini(test_messages)
    print_hl(chat_gemini.__name__ + " >")
    print(rst)
    rst = chat_gemini_flash(test_messages)
    print_hl(chat_gemini_flash.__name__ + " >")
    print(rst)
    # rst = chat_t3v(test_messages)
    # print(rst)
