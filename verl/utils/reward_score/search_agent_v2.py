"""
search_agent_v2: Alignment-only reward extension for search-agent geolocation.

Based on utils/reward_score/search_agent.py (base score), plus an alignment reward:
- If any search result payload mentions the target city and the model's final
  answer is city-correct → +1.0
- If any search result payload mentions the target city but the final answer is
  not city-correct → -1.0

All helper parsing remains local to this file, avoiding side effects.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .utils_geo import eval_geolocation_response
from .utils import print_hl, print_error


# -------- Helpers: extract search results from conversation text -------- #

_TOOL_RESP_RE = re.compile(r"<tool_response>([\s\S]*?)</tool_response>", re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.IGNORECASE | re.MULTILINE)


def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    if not text:
        return objs
    s = text
    n = len(s)
    i = 0
    while i < n:
        if s[i] == '{':
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                ch = s[j]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                j += 1
            if depth == 0:
                cand = s[i:j]
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict):
                        objs.append(obj)
                except Exception:
                    pass
                i = j
                continue
        i += 1
    return objs


def _extract_tool_responses(text: str) -> List[str]:
    return [m.group(1) for m in _TOOL_RESP_RE.finditer(text or "")]


def _extract_search_result_payloads(text: str) -> List[Dict[str, Any]]:
    """Return objects shaped like {"name":"search_web", "query":..., "result": ...}
    that appear inside <tool_response> blocks.
    """
    payloads: List[Dict[str, Any]] = []
    for seg in _extract_tool_responses(text):
        for obj in _extract_json_objects(seg):
            try:
                if obj.get("name") == "search_web" and "result" in obj:
                    payloads.append(obj)
            except Exception:
                continue
    return payloads


def _extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool-call dicts, ignoring any calls echoed inside <tool_response> blocks."""
    calls: List[Dict[str, Any]] = []
    text = text or ""
    resp_spans: List[Tuple[int, int]] = []
    for rm in _TOOL_RESP_RE.finditer(text):
        resp_spans.append((rm.start(), rm.end()))

    def _inside_resp(idx: int) -> bool:
        for s, e in resp_spans:
            if s <= idx < e:
                return True
        return False

    for m in _TOOL_CALL_RE.finditer(text):
        if _inside_resp(m.start()):
            continue
        raw = m.group(1)
        for obj in _extract_json_objects(raw):
            if isinstance(obj, dict):
                calls.append(obj)
    return calls


def _extract_search_queries(text: str) -> List[str]:
    queries: List[str] = []
    seen: set = set()
    for call in _extract_tool_calls(text):
        try:
            name = str(call.get("name") or "").strip()
            if name != "search_web":
                continue
            args = call.get("arguments") or {}
            q = args.get("query")
            if isinstance(q, str):
                q2 = q.strip()
                if q2 and q2 not in seen:
                    queries.append(q2)
                    seen.add(q2)
        except Exception:
            continue
    return queries


# ---- City-name/template detection for anchored@k ---- #

_KNOWN_CITIES = {
    # English
    "new york", "london", "paris", "tokyo", "beijing", "shanghai", "los angeles",
    "san francisco", "chicago", "berlin", "madrid", "rome", "milan", "barcelona",
    "sydney", "melbourne", "toronto", "vancouver", "dubai", "mumbai", "delhi",
    "bangkok", "kuala lumpur", "seoul", "busan", "osaka", "kyoto", "amsterdam",
    "rotterdam", "brussels", "vienna", "munich", "hamburg", "frankfurt", "zurich",
    "geneva", "prague", "warsaw", "lisbon", "athens", "istanbul", "cairo",
    "johannesburg", "mexico city", "sao paulo", "rio de janeiro", "buenos aires",
    "santiago",
    # Chinese
    "纽约", "伦敦", "巴黎", "东京", "北京", "上海", "深圳", "广州", "成都", "重庆", "香港", "新加坡",
    "洛杉矶", "旧金山", "芝加哥", "柏林", "马德里", "罗马", "米兰", "巴塞罗那", "悉尼", "墨尔本",
    "多伦多", "温哥华", "迪拜", "孟买", "德里", "曼谷", "吉隆坡", "首尔", "釜山", "大阪", "京都",
    "阿姆斯特丹", "鹿特丹", "布鲁塞尔", "维也纳", "慕尼黑", "汉堡", "法兰克福", "苏黎世", "日内瓦",
    "布拉格", "华沙", "里斯本", "雅典", "伊斯坦布尔", "开罗", "约翰内斯堡", "墨西哥城", "圣保罗",
    "里约热内卢", "布宜诺斯艾利斯", "圣地亚哥",
}

_CITY_TEMPLATES = {
    # English
    "eiffel tower", "big ben", "statue of liberty", "times square", "golden gate bridge",
    "sydney opera house", "burj khalifa", "colosseum", "sagrada familia", "tower bridge",
    "shibuya crossing", "forbidden city", "marina bay sands", "merlion", "cn tower",
    "petronas towers", "brandenburg gate", "acropolis", "christ the redeemer",
    "red square", "st basil", "duomo di milano", "la rambla",
    # Chinese
    "埃菲尔铁塔", "大本钟", "自由女神像", "时代广场", "金门大桥", "悉尼歌剧院", "哈利法塔", "迪拜塔", "罗马斗兽场",
    "圣家堂", "伦敦塔桥", "涩谷十字路口", "故宫", "紫禁城", "滨海湾金沙", "鱼尾狮", "加拿大国家电视塔",
    "双子塔", "双峰塔", "勃兰登堡门", "卫城", "救世基督像", "红场", "圣瓦西里", "米兰大教堂", "兰布拉大道",
}


def _lower(s: str) -> str:
    try:
        return s.lower()
    except Exception:
        return s


def _find_city_mentions(query: str, ground_truth: Dict[str, Any]) -> bool:
    ql = _lower(query)
    # gold location tokens provide an easy anchor if present
    for key in ("city", "province_or_state", "state", "country"):
        val = ground_truth.get(key)
        if isinstance(val, str) and val.strip():
            v = _lower(val.strip())
            if v and v in ql:
                return True
    for city in _KNOWN_CITIES:
        if city in ql:
            return True
    for tmpl in _CITY_TEMPLATES:
        if tmpl in ql:
            return True
    return False


def _contains_ci(haystack: str, needle: str) -> bool:
    try:
        return needle.lower() in haystack.lower()
    except Exception:
        return needle in haystack


def _search_results_contain_target_city(payloads: List[Dict[str, Any]], ground_truth: Dict[str, Any]) -> bool:
    city = (ground_truth.get("city") or "").strip()
    if not city:
        return False
    # Concatenate payloads into a single string and check substring (case-insensitive)
    try:
        buf = "\n".join(json.dumps(p, ensure_ascii=False) for p in payloads)
    except Exception:
        buf = "\n".join(str(p) for p in payloads)
    return _contains_ci(buf, city)


# ------------------------------
# Main reward computation
# ------------------------------

def compure_score_search(predict_str: str, ground_truth: dict, extra_info=None) -> Dict[str, Any]:
    """Base geolocation score + alignment reward.

    Returns a dict with keys:
    - score: 0/1/2/4 base correctness (city/state/country)
    - city_correct/state_correct/country_correct: floats (0.0/1.0)
    - alignment_reward: +/- reward based on search results vs final city correctness
    - search_contains_target_city: 1.0 if target city appears in search results, else 0.0
    """
    raw_text = predict_str or ""

    # Evaluate final answer correctness (use last assistant chunk as in v0)
    last_predict_str = raw_text.split('assistant')[-1].strip()
    combined = eval_geolocation_response(
        last_predict_str,
        ground_truth,
        model_verifier=True,
        api_key='307ad8bb-2c44-4911-85e8-48ea32f6a672',
    )

    if combined['city_correct']:
        base_score = 4.0
    elif combined['state_correct']:
        base_score = 2.0
    elif combined['country_correct']:
        base_score = 1.0
    else:
        base_score = 0.0


    # Alignment reward
    payloads = _extract_search_result_payloads(raw_text)
    search_has_target_city = _search_results_contain_target_city(payloads, ground_truth)
    if search_has_target_city and combined['city_correct']:
        align_reward = 1.0
    elif search_has_target_city and not combined['city_correct']:
        align_reward = -1.0
    else:
        align_reward = 0.0

    # Anchored metrics from the first two search queries
    queries = _extract_search_queries(raw_text)
    anchored_bits: List[float] = []
    for q in queries[:2]:
        anchored_bits.append(1.0 if _find_city_mentions(q, ground_truth) else 0.0)

    return {
        'score': base_score,
        'city_correct': 1.0 if combined.get('city_correct') else 0.0,
        'state_correct': 1.0 if combined.get('state_correct') else 0.0,
        'country_correct': 1.0 if combined.get('country_correct') else 0.0,
        'alignment_reward': float(align_reward),
        'search_contains_target_city': 1.0 if search_has_target_city else 0.0,
        'anchored@0': float(anchored_bits[0]) if len(anchored_bits) > 0 else 0.0,
        'anchored@1': float(anchored_bits[1]) if len(anchored_bits) > 1 else 0.0,
    }


if __name__ == "__main__":
    fake_conv = (
        "<|im_start|>assistant\n"
        "<tool_response>For search, the search results are as follows:\n"
        "{\n  \"name\": \"search_web\",\n  \"query\": \"Sunglass Hut near Paris\",\n  \"result\": [{\"title\": \"Store Paris\", \"snippet\": \"...Paris...\"}]\n}\n"
        "</tool_response>\n"
        "Final guess: Paris, France."
    )
    gt = {"country": "France", "province_or_state": None, "city": "Paris"}
    out = compure_score_search(fake_conv, gt)
    print_hl(json.dumps(out, ensure_ascii=False))
