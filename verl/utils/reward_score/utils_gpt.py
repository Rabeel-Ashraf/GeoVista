# eval/utils_gemini.py
import os
import io
import base64
import imghdr
import requests
from PIL import Image
import time
import sys
from .utils import print_error, print_hl

def _base64_encode_image(image_file: str):
    """
    支持本地路径或 http(s) URL。
    返回 (base64_str, image_format)；image_format 例如 'jpeg', 'png'
    """
    if image_file.startswith('http://') or image_file.startswith('https://'):
        resp = requests.get(image_file, timeout=30)
        resp.raise_for_status()
        image_bytes = resp.content
        image_format = imghdr.what(None, h=image_bytes) or 'jpeg'
    else:
        img = Image.open(image_file)
        img = img.convert('RGB')
        w, h = img.size
        max_size = 2048
        if w * h > max_size * max_size:
            scale = (max_size * max_size / (w * h)) ** 0.5
            img = img.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        image_bytes = buf.getvalue()
        image_format = 'jpeg'
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_b64, image_format

def _to_proxy_message_content(messages):
    """
    将 eval_mat 风格的输入转换为后端期望的:
    [{"role":"user","content":[{"type":"image_url","value":"data:..."}, {"type":"text","value":"..."}]}]
    同时抽取 system prompt 字符串。
    """
    user_content = []
    system_prompt = ""

    for m in messages:
        role = m.get('role')
        content = m.get('content')

        if role == 'system':
            if isinstance(content, str):
                system_prompt += content
            elif isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        system_prompt += item.get('text', '') or item.get('value', '') or ''
            continue

        if role == 'user' and isinstance(content, list):
            for item in content:
                t = item.get('type')
                if t == 'text':
                    text_val = item.get('text') or item.get('value') or ''
                    if text_val:
                        user_content.append({"type": "text", "value": text_val})
                elif t == 'image_url':
                    # 兼容 {"type":"image_url","image_url":{"url": ...}} 或 {"type":"image_url","url": ...} 或 {"type":"image_url","value": ...}
                    url = None
                    if isinstance(item.get('image_url'), dict):
                        url = item['image_url'].get('url')
                    url = url or item.get('url') or item.get('value')

                    if not url:
                        continue

                    if url.startswith('data:image/'):
                        data_url = url
                    else:
                        b64, fmt = _base64_encode_image(url)
                        data_url = f"data:image/{fmt.lower()};base64,{b64}"

                    user_content.append({"type": "image_url", "value": data_url})

    if not user_content:
        # 至少保证有一个空文本，防止后端报错
        user_content.append({"type": "text", "value": ""})

    return [{"role": "user", "content": user_content}], system_prompt

def chat_gemini(
    messages,
    model_name: str = "api_openai_gpt-5",
    model_marker: str = "api_openai_gpt-5",
    api_key: str = None,
    timeout: int = 300,
    api_url: str = "http://trpc-utools-prod.turbotke.production.polaris:8009/",
):
    """
    参数:
    - messages: 按 eval_mat 风格的图文交错 message 列表
    - api_key: 不提供则从环境变量 HY_API_KEY 或 API_KEY 读取
    返回:
    - str (模型回答)
    """
    proxy_messages, system_prompt = _to_proxy_message_content(messages)
    # assert model_name in ["api_openai_gpt-5", "azure-gpt-4o-mini"]
    
    # if model_marker == "api_azure_openai_gpt-4o-mini"
    if model_name == 'azure-gpt-4o-mini':
        model_marker = "api_azure_openai_gpt-4o-mini"
    else:
        model_marker = model_name
    payload = {
        "bid": "open_api_test",
        "server": "open_api",
        "services": [],
        "request_id": "1234",
        "session_id": "12345",
        "api_key": api_key,
        "model_marker": model_marker,
        "system": system_prompt or "",
        "params": {},
        "timeout": 100,
        "model_name": model_name,
        "messages": proxy_messages,
    }


    resp = requests.post(api_url, json=payload, timeout=timeout, proxies={"http": None, "https": None})
    resp.raise_for_status()
    data = resp.json()

    retry = 5
    sleep_time = 2
    while retry:
        try:
            return data["answer"][0]["value"]
        except Exception:
            # 兜底返回完整响应，便于调试
            print_error(data)
            # buffer instant output
            sys.stdout.flush()
            retry -= 1
            time.sleep(sleep_time)
            sleep_time *= 10
            sleep_time = min(sleep_time, 300)

    raise ValueError(f"Failed to get answer from GPT-5: {data}")


def chat_4o_mini(messages, api_key=None, timeout=30) -> str:
    api_key = os.environ.get("GPT_CUSTOM_API_KEY_4o_mini")
    if not api_key:
        raise ValueError("api_key is required. Set GPT_CUSTOM_API_KEY_4o_mini in env, or pass api_key explicitly.")
    return chat_gemini(messages, model_name="azure-gpt-4o-mini", api_key=api_key, timeout=timeout)


def chat_gpt5_unstable(messages, api_key=None, timeout=30) -> str:
    api_key = os.environ.get("GPT_CUSTOM_API_KEY")
    if not api_key:
        raise ValueError("api_key is required. Set GPT_CUSTOM_API_KEY in env, or pass api_key explicitly.")
    return chat_gemini(messages, model_name="api_openai_gpt-5", api_key=api_key, timeout=timeout)


def chat_gpt5(messages, api_key=None, timeout=30) -> str:
    return chat_gpt5_chat(messages, api_key=api_key, timeout=timeout)


def chat_gpt5_chat(messages, api_key=None, timeout=30) -> str:
    api_key = os.environ.get("GPT5_CHAT_CUSTOM_API_KEY")
    if not api_key:
        raise ValueError("api_key is required. Set GPT5_CHAT_CUSTOM_API_KEY in env, or pass api_key explicitly.")
    return chat_gemini(messages, model_name="api_azure_openai_gpt-5-chat-latest", api_key=api_key, timeout=timeout)


def chat_doubao(messages, timeout=30) -> str:
    api_key = os.environ.get("DOUBAO_CUSTOM_API_KEY")
    if not api_key:
        raise ValueError("api_key is required. Set DOUBAO_CUSTOM_API_KEY in env, or pass api_key explicitly.")
    return chat_gemini(messages, model_name="api_doubao_DouBao-Seed-1.6-Vision-250815", api_key=api_key, timeout=timeout)


if __name__ == '__main__':
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                # {"type": "image_url", "image_url": {"url": "/Users/mansionchieng/Workspaces/agent_workstation/hunyuan-o3/.temp/ipynb/outputs/qwen/img/5929fbb5e2f844ca955c2aeabbc99ab6.jpg"}},
                {"type": "text", "text": "Please describe this picture."}
            ],
        },
    ]
    print_hl("[TEST] chat_4o_mini")
    print(chat_4o_mini(messages))
    # print_hl("[TEST] chat_doubao")
    # print(chat_doubao(messages))
    # print_hl("[TEST] chat_gpt5_chat")
    # print(chat_gpt5_chat(messages))
    # print_hl("[TEST] chat_gpt5_unstable")
    # print(chat_gpt5_unstable(messages))
    # print_hl("[TEST] chat_gpt5")
    # print(chat_gpt5(messages))
