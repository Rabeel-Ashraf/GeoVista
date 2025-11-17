"""
Enhanced search-agent reward with query and alignment components.

Based on utils/reward_score/search_agent.py but adds:
- Query reward: penalize queries that inject city names or use "famous city templates".
- Alignment reward: if search results contain the target city and the model's
  final answer is correct (city-level), reward; if incorrect, penalize.

All helpers are kept within this file to avoid affecting other modules.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .utils_geo import eval_geolocation_response
from .utils import print_hl, print_error


# ------------------------------
# Helpers: text + conversation parsing
# ------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.IGNORECASE | re.MULTILINE)
_TOOL_RESP_RE = re.compile(r"<tool_response>([\s\S]*?)</tool_response>", re.IGNORECASE)


def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Extract well-formed JSON dicts from arbitrary text using a brace counter.
    Returns a list of dicts; invalid parses are skipped.
    """
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


def _extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract structured tool-call dicts, skipping duplicates echoed inside <tool_response> blocks.

    Some pipelines echo the same <tool_call> JSON inside a <tool_response> message
    (e.g., "For <tool_call>{...}</tool_call>, the results are ..."), which would double count.
    We avoid this by ignoring any <tool_call> spans that fall within a <tool_response> span.
    """
    calls: List[Dict[str, Any]] = []
    text = text or ""

    # Collect <tool_response> spans
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
        objs = _extract_json_objects(raw)
        for obj in objs:
            if isinstance(obj, dict):
                calls.append(obj)
    return calls


def _extract_tool_responses(text: str) -> List[str]:
    return [m.group(1) for m in _TOOL_RESP_RE.finditer(text or "")]


def _extract_search_queries(text: str) -> List[str]:
    """Return queries from tool calls of name 'search_web'."""
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


def _extract_search_result_payloads(text: str) -> List[Dict[str, Any]]:
    """Parse JSON payloads that look like {"name":"search_web", "query":..., "result": ...}
    from inside <tool_response> blocks.
    """
    payloads: List[Dict[str, Any]] = []
    for seg in _extract_tool_responses(text):
        for obj in _extract_json_objects(seg):
            try:
                # keep only the actual results payload (has 'result'), not the echoed tool_call (has 'arguments')
                if obj.get("name") == "search_web" and "result" in obj:
                    payloads.append(obj)
            except Exception:
                continue
    return payloads


# ------------------------------
# City/token detection
# ------------------------------

# A compact list of well-known cities in EN/ZH to catch template-style queries.
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

# City-anchored landmark templates (知名城市模板) that strongly imply guessing a city by template.
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


def _contains_any(haystack: str, needles: List[str]) -> bool:
    h = _lower(haystack)
    for n in needles:
        n2 = _lower(n)
        if n2 and n2 in h:
            return True
    return False


def _find_city_mentions(query: str, ground_truth: Dict[str, Any]) -> List[str]:
    """Return a list of matched city-like tokens found in query.
    Includes the gold city/state/country and a set of well-known cities/templates.
    """
    hits: List[str] = []
    ql = _lower(query)

    # gold terms
    for key in ("city", "province_or_state", "state", "country"):
        val = ground_truth.get(key)
        if isinstance(val, str) and val.strip():
            v = _lower(val.strip())
            if v and v in ql:
                hits.append(f"gt:{val}")

    # known cities
    for city in _KNOWN_CITIES:
        if city in ql:
            hits.append(f"city:{city}")

    # landmark templates
    for tmpl in _CITY_TEMPLATES:
        if tmpl in ql:
            hits.append(f"tmpl:{tmpl}")

    # Basic language patterns like "in <city>" are implicitly covered by above sets.
    return hits


def _search_results_contain_target_city(payloads: List[Dict[str, Any]], ground_truth: Dict[str, Any]) -> bool:
    city = _norm(ground_truth.get("city"))
    if not city:
        return False

    needles = [city]
    # also allow simple variations like city with comma or parentheses to be matched as substring
    buf_parts: List[str] = []
    for p in payloads:
        try:
            buf_parts.append(json.dumps(p, ensure_ascii=False))
        except Exception:
            buf_parts.append(str(p))
    hay = "\n" + "\n".join(buf_parts)
    return _contains_any(hay, needles)


# ------------------------------
# Main reward computation
# ------------------------------

def compure_score_search(predict_str: str, ground_truth: dict, extra_info=None) -> Dict[str, Any]:
    """Compute base geolocation score + query and alignment rewards.

    Returns a dict with keys:
    - score: 0/1/2/4 as base correctness (city/state/country)
    - city_correct/state_correct/country_correct: floats (0.0/1.0)
    - query_reward: negative penalty for city-injected or template queries
    - alignment_reward: +/- reward based on search results vs final correctness
    - diagnostics (optional): details for analysis
    """
    raw_text = predict_str or ""

    # 1) Evaluate final answer correctness using the last assistant chunk
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

    # 2) Query reward: penalize city-name injections or famous-city templates
    offending: List[Tuple[str, List[str]]] = []  # (query, hits)
    queries = _extract_search_queries(raw_text)
    anchored_bits: List[float] = []  # 1.0 if the query contains any city/template/gold term
    for q in queries:
        hits = _find_city_mentions(q, ground_truth)
        anchored_bits.append(1.0 if len(hits) > 0 else 0.0)
        if hits:
            offending.append((q, hits))

    # Penalty policy: -0.5 per offending query (cap at -2.0)
    query_penalty = -0.5 * len(offending)
    if query_penalty < -2.0:
        query_penalty = -2.0

    # 3) Alignment reward: if search results contain target city
    #    - and city_correct=True  -> +1.0
    #    - and city_correct=False -> -1.0
    payloads = _extract_search_result_payloads(raw_text)
    search_has_target_city = _search_results_contain_target_city(payloads, ground_truth)
    if search_has_target_city and combined['city_correct']:
        align_reward = 1.0
    elif search_has_target_city and not combined['city_correct']:
        align_reward = -1.0
    else:
        align_reward = 0.0

    score = base_score + query_penalty + align_reward

    result = {
        'score': score,
        'base_score': base_score,
        'city_correct': 1.0 if combined.get('city_correct') else 0.0,
        'state_correct': 1.0 if combined.get('state_correct') else 0.0,
        'country_correct': 1.0 if combined.get('country_correct') else 0.0,
        # new components
        'query_reward': float(query_penalty),
        'alignment_reward': float(align_reward),
        # lightweight diagnostics (kept minimal to avoid breaking callers)
        'num_search_calls': int(len(queries)),
        'search_contains_target_city': 1.0 if search_has_target_city else 0.0,
        # anchored flags for first and second search queries (if any)
        'anchored@0': float(anchored_bits[0]) if len(anchored_bits) > 0 else 0.0,
        'anchored@1': float(anchored_bits[1]) if len(anchored_bits) > 1 else 0.0,
    }

    # Optionally attach detailed offending queries for debugging
    try:
        if offending:
            result['offending_queries'] = [
                {'query': q, 'hits': hits[:8]} for q, hits in offending[:6]
            ]
    except Exception:
        pass

    return result


if __name__ == "__main__":
    # Simple smoke test for the new parser and reward components
    fake_conv = (
        "<|im_start|>user\nPlease search.\n<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<tool_call>{\n  \"name\": \"search_web\",\n  \"arguments\": {\"query\": \"Sunglass Hut near Paris\"}\n}</tool_call>\n"
        "<tool_response>For search, the search results are as follows:\n"
        "{\n  \"name\": \"search_web\",\n  \"query\": \"Sunglass Hut near Paris\",\n  \"result\": [{\"title\": \"Store Paris\", \"snippet\": \"...Paris...\"}]\n}\n"
        "</tool_response>\n"
        "Final guess: Paris, France."\n
    )
    gt = {"country": "France", "province_or_state": None, "city": "Paris"}
    out = compure_score_search(fake_conv, gt)
    print_hl(json.dumps(out, ensure_ascii=False))
