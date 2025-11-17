# eval/multi_level_eval_gemini_0825.py
import json
import re
import unicodedata
from typing import Dict, Optional, Tuple, Any

# from utils_api import chat_gemini
# from utils_gpt import chat_4o_mini
from utils_api import chat_4o_mini, chat_gemini
from utils import print_hl


def _normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _contains_whole_term(text_norm: str, term_norm: str) -> bool:
    if not term_norm:
        return False
    pattern = r"(?<!\w)" + re.escape(term_norm) + r"(?!\w)"
    return re.search(pattern, text_norm) is not None


def _extract_json_obj(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s = s.strip()
    # Try direct parse first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fallback: extract first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = s[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _read_possible_lat_lng(d: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if not isinstance(d, dict):
        return None
    cand_lat_keys = ["lat", "latitude"]
    cand_lng_keys = ["lng", "lon", "long", "longitude"]
    lat = None
    lng = None
    for k in cand_lat_keys:
        if k in d and d[k] is not None:
            lat = d[k]
            break
    for k in cand_lng_keys:
        if k in d and d[k] is not None:
            lng = d[k]
            break
    if lat is None or lng is None:
        return None
    try:
        return float(lat), float(lng)
    except Exception:
        return None


def _read_possible_text_fields(d: Dict[str, Any]) -> Dict[str, Optional[str]]:
    if not isinstance(d, dict):
        return {"country": None, "province_or_state": None, "city": None}

    country_keys = ["country", "country_name", "nation"]
    state_keys = [
        "province_or_state",
        "state",
        "province",
        "state_province",
        "admin1",
        "region",
        "state_name",
        "province_name",
    ]
    city_keys = ["city", "city_name", "town", "locality", "admin2", "county"]

    def pick(keys):
        for k in keys:
            if k in d and d[k]:
                return str(d[k])
        return None

    return {
        "country": pick(country_keys),
        "province_or_state": pick(state_keys),
        "city": pick(city_keys),
    }


def loc_convert_dict(
    location: Dict[str, Any],
    use_model: bool = True,
    api_key: Optional[str] = None,
    timeout: int = 120,
    meta_city: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    将 metadata/location 转换成:
    {
        "country": str or None,
        "province_or_state": str or None,
        "city": str or None,
        "gemini_remark": str or None
    }

    规则:
    - 优先使用 meta_city（如无则回退到 location 中的 city）。
    - 使用 Gemini 仅补全缺失层级；当没有省级/州级（如 Shanghai），将 province_or_state 设为 city；
      当 city 与国家一致（如 Singapore），三级都设为同一值。
    - 强健错误处理：当 Gemini 返回不可解析或字段不全时，使用更严格提示重试（最多 3 次）。
    - 生成后，调用 Gemini 基于经纬度与三层行政区信息做一致性检查，返回简短 remark。
    """
    import json as _json

    def _norm(x: Optional[str]) -> str:
        import unicodedata, re
        if not x:
            return ""
        s = unicodedata.normalize("NFKD", str(x))
        s = s.encode("ascii", "ignore").decode("ascii")
        s = s.lower()
        s = re.sub(r"[^a-z0-9\\s]", " ", s)
        s = re.sub(r"\\s+", " ", s).strip()
        return s

    base = _read_possible_text_fields(location)
    latlng = _read_possible_lat_lng(location)  # Optional[Tuple[float, float]]

    city = (meta_city or base.get("city")) or None
    country = base.get("country")
    state = base.get("province_or_state")

    # If all present already, still produce remark based on lat/lng if any.
    need_country = not country
    need_state = not state

    def _need_completion(c: Optional[str], s: Optional[str], co: Optional[str]) -> bool:
        return (not co) or (not s) or (not c)

    def _try_parse_complete_obj(obj: Any) -> Optional[Dict[str, Optional[str]]]:
        if not isinstance(obj, dict):
            return None
        co = obj.get("country"); st = obj.get("province_or_state") or obj.get("state") or obj.get("admin1"); ci = obj.get("city")
        # Accept partial JSON (we will post-process), but ensure dict.
        return {"country": str(co) if co else None,
                "province_or_state": str(st) if st else None,
                "city": str(ci) if ci else None}

    def _reactive_messages(task_payload: Dict[str, Any], stricter: int) -> list:
        sys_common = (
            "You are a precise reverse-geocoding assistant. "
            "Output JSON only, with keys: country, province_or_state, city. "
            "Use official English names. If a field is truly unknown, set it to null. "
            "Do not add any extra keys or text."
        )
        extra_rules = []
        # 规则：city=state 或 city=country 允许；新加坡等城邦国家三级可相同。
        extra_rules.append("If a city is also a province/state (e.g., Shanghai), set province_or_state equal to the city.")
        extra_rules.append("If the city equals the country (e.g., Singapore), set all three fields to that country/city name.")
        if stricter >= 1:
            extra_rules.append("Your previous result was invalid or incomplete. Ensure all three keys are present in the JSON. Use null only if truly unknown.")
        if stricter >= 2:
            extra_rules.append("Be concise and strictly return a valid JSON object with exactly the three required keys and string/null values.")
        sys_prompt = sys_common + " " + " ".join(extra_rules)
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [{"type": "text", "text": _json.dumps(task_payload, ensure_ascii=False)}]},
        ]

    def _complete_with_gemini(existing_city: Optional[str],
                              existing_state: Optional[str],
                              existing_country: Optional[str]) -> Dict[str, Optional[str]]:
        if not use_model:
            return {"country": existing_country, "province_or_state": existing_state, "city": existing_city}
        payload: Dict[str, Any] = {
            "task": "complete_location_hierarchy",
            "need_fields": ["country", "province_or_state", "city"],
            "hints": {
                "city": existing_city,
                "province_or_state": existing_state,
                "country": existing_country,
            },
        }
        if latlng is not None:
            payload["latitude"] = latlng[0]
            payload["longitude"] = latlng[1]

        last_err: Optional[str] = None
        for attempt in range(3):
            try:
                messages = _reactive_messages(payload, stricter=attempt)
                # resp = chat_gemini(messages, api_key=api_key, timeout=timeout)
                resp = chat_gemini(messages)
                obj = _extract_json_obj(resp)
                parsed = _try_parse_complete_obj(obj)
                if not parsed:
                    last_err = "parse_error"
                    continue

                # Merge with existing values, keep existing city if provided.
                out_city = existing_city or parsed.get("city")
                out_country = existing_country or parsed.get("country")
                out_state = existing_state or parsed.get("province_or_state")

                # Post rules for city/state/country equality.
                if out_city:
                    # If state missing, allow state = city
                    if not out_state:
                        out_state = out_city
                    # If country equals city or missing:
                    if not out_country or _norm(out_country) == _norm(out_city):
                        out_country = out_city
                        out_state = out_city

                return {
                    "country": out_country,
                    "province_or_state": out_state,
                    "city": out_city,
                }
            except Exception as e:
                if 'GEMINI_CUSTOM_API_KEY' in str(e):
                    raise e
                last_err = f"exception:{e}"
                print(last_err)
                continue

        # Fallback without model or after failures:
        out_city = existing_city
        out_state = existing_state
        out_country = existing_country
        if out_city:
            if not out_state:
                out_state = out_city
            if not out_country:
                out_country = out_city
        return {"country": out_country, "province_or_state": out_state, "city": out_city}

    # Complete if needed
    # if _need_completion(city, state, country):
    completed = _complete_with_gemini(city, state, country)
    country, state, city = completed["country"], completed["province_or_state"], completed["city"]
    # else:
    #     # Even if complete, normalize equality rules
    #     if city and (not state):
    #         state = city
    #     if city and (not country or _norm(country) == _norm(city)):
    #         country = city
    #         state = city

    # Create remark by asking Gemini to check lat/lng vs labels
    def _remark_messages(latlng: Optional[tuple], co: Optional[str], st: Optional[str], ci: Optional[str], strict: int) -> list:
        sys_prompt = (
            "You are a location validator. Given latitude/longitude and a 3-level admin label "
            "(country, province_or_state, city), output JSON only with key: remark. "
            "Remark should be a short English sentence describing whether the coordinates match the labels, "
            "e.g., 'match', or 'mismatch: reason'. Consider city-states like Singapore and cases where city equals state."
        )
        if strict >= 1:
            sys_prompt += " Your previous response was invalid. Return JSON like {\"remark\": \"...\"} only."
        user_payload: Dict[str, Any] = {
            "labels": {"country": co, "province_or_state": st, "city": ci},
        }
        if latlng is not None:
            user_payload["latitude"] = latlng[0]
            user_payload["longitude"] = latlng[1]
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [{"type": "text", "text": _json.dumps(user_payload, ensure_ascii=False)}]},
        ]

    gemini_remark: Optional[str] = None
    if use_model and (latlng is not None or city or country or state):
        for attempt in range(3):
            try:
                msgs = _remark_messages(latlng, country, state, city, strict=attempt)
                # resp = chat_gemini(msgs, api_key=api_key, timeout=timeout)
                resp = chat_gemini(msgs)
                obj = _extract_json_obj(resp)
                if isinstance(obj, dict) and isinstance(obj.get("remark"), str):
                    gemini_remark = obj["remark"]
                    break
            except Exception:
                continue
        if not gemini_remark:
            gemini_remark = "no_remark: validator failed"
    else:
        gemini_remark = "no_remark: model disabled or insufficient inputs"

    return {
        "country": country,
        "province_or_state": state,
        "city": city,
        "gemini_remark": gemini_remark,
    }
# ... other helpers remain unchanged ...


def eval_geolocation_response(
    response: str,
    loc_dict: Dict[str, Optional[str]],
    model_verifier: bool = False,
    api_key: Optional[str] = None,
    timeout: int = 120,
    debug_mode: bool = False,
) -> Dict[str, Any]:
    """
    评测模型回答的地理定位是否与 loc_dict 一致。
    流程：
    1) 先进行本地规则匹配（rule-based）得到 rb = {country_correct, state_correct, city_correct}。
    2) 若 model_verifier=True，则调用 Gemini 进行一次判断，得到 mb，同样三个布尔值；
       然后与 rb 做按位“或”合并：combined = rb OR mb。
    3) 对 combined 做阶梯式一致性修正：
       - 若 city_correct=True，则同时令 state_correct=True, country_correct=True。
       - 否则若 state_correct=True，则令 country_correct=True。
    4) 返回修正后的 combined。
    """

    # ------ 规则匹配（本地）------
    def rule_based() -> Dict[str, bool]:
        res_norm = _normalize_text(response or "")
        city_norm = _normalize_text(loc_dict.get("city"))
        state_norm = _normalize_text(
            loc_dict.get("province_or_state") or loc_dict.get("state")
        )
        country_norm = _normalize_text(loc_dict.get("country"))

        city_hit = _contains_whole_term(res_norm, city_norm) if city_norm else False
        state_hit = _contains_whole_term(res_norm, state_norm) if state_norm else False
        country_hit = _contains_whole_term(res_norm, country_norm) if country_norm else False

        city_correct = city_hit
        state_correct = state_hit or city_hit
        country_correct = country_hit or state_hit or city_hit

        return {
            "country_correct": bool(country_correct),
            "state_correct": bool(state_correct),
            "city_correct": bool(city_correct),
        }

    rb = rule_based()  # 1) 始终先做本地规则匹配
    print_hl("[eval_geolocation_response] rule based raw response:")
    print(rb if isinstance(rb, str) else json.dumps(rb, ensure_ascii=False))

    # 若不启用模型裁判，直接做阶梯一致性并返回
    if not model_verifier:
        combined = dict(rb)
        # 3) 阶梯式一致性修正
        if combined["city_correct"]:
            combined["state_correct"] = True
            combined["country_correct"] = True
        elif combined["state_correct"]:
            combined["country_correct"] = True
        return combined

    # ------ 模型裁判（一次调用）------
    def _model_verdict() -> Optional[Dict[str, bool]]:
        try:
            sys_prompt = (
                "You are a strict evaluator. Decide if a free-text geolocation answer matches a gold location.\n"
                "Rules:\n"
                "1) If the answer names the correct city (as a toponym), then city/state/country are all True.\n"
                "2) If it names the correct state/province (but not the correct city), then state and country are True; city is False.\n"
                "3) If it names only the correct country, then only country is True.\n"
                "4) If none match, all are False.\n"
                "5) Consider common synonyms and English exonyms; ignore punctuation and case.\n"
                "Respond with a single JSON object: {\"country_correct\": <bool>, \"state_correct\": <bool>, \"city_correct\": <bool>}."
            )
            user_payload = {
                "gold_location": {
                    "country": loc_dict.get("country"),
                    "province_or_state": loc_dict.get("province_or_state") or loc_dict.get("state"),
                    "city": loc_dict.get("city"),
                },
                "model_response": response,
            }
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}]},
            ]
            resp = chat_4o_mini(messages, timeout=timeout)
            if debug_mode and model_verifier:
                try:
                    print_hl("[eval_geolocation_response] model_verifier raw response:")
                    print(resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False))
                except Exception:
                    pass
            obj = _extract_json_obj(resp)
            if isinstance(obj, dict):
                return {
                    "country_correct": bool(obj.get("country_correct") is True),
                    "state_correct": bool(obj.get("state_correct") is True),
                    "city_correct": bool(obj.get("city_correct") is True),
                }
            else:
                raise ValueError(f'Could not extract json object from raw response: {resp}')
        except Exception as e:
            raise e

    mb = _model_verdict()

    # 2) model verifier 为准
    combined = mb

    # 3) 阶梯式一致性修正
    if combined["city_correct"]:
        combined["state_correct"] = True
        combined["country_correct"] = True
    elif combined["state_correct"]:
        combined["country_correct"] = True

    # 4) 返回
    return combined
