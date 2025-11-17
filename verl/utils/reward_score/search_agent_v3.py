# eval/multi_level_eval_gemini_0825.py
import json
from typing import Dict, Optional, Tuple, Any

from .utils_geo import eval_geolocation_response
from .utils import print_hl, print_error


def compure_score_search(predict_str: str, ground_truth: dict, extra_info=None) -> float:
    last_predict_str = predict_str.split('assistant')[-1].strip()
    # print(f'[DEBUG EVALUATION] compute_score_search \n{predict_str=} \n{ground_truth=} \n{last_predict_str=}')

    combined = eval_geolocation_response(last_predict_str, ground_truth, 
                                         model_verifier=True,
                                         api_key='307ad8bb-2c44-4911-85e8-48ea32f6a672')
    # if combined['city_correct']:
    #     score = 4.
    # elif combined['state_correct']:
    #     score = 2.
    # elif combined['country_correct']:
    #     score = 1.
    if combined['country_correct']:
        score = 1.
    else:
        score = 0.
    
    return {
        'score': score,
        'city_correct': 1.0 if combined['city_correct'] else 0.0,
        'state_correct': 1.0 if combined['state_correct'] else 0.0,
        'country_correct': 1.0 if combined['country_correct'] else 0.0,
    }


if __name__ == "__main__":
    # Lightweight tests for compure_score_search with a stubbed verifier
    # to avoid network calls. Each case overrides chat_4o_mini to return
    # a controlled JSON verdict and prints the computed score mapping.

    def run_case(name: str, stub_verdict: Dict[str, bool], predict_str: str, loc: Dict[str, str]):
        # Monkeypatch the verifier
        global chat_4o_mini
        _orig_chat = chat_4o_mini

        def _fake_chat(messages, timeout: int = 120, api_key: Optional[str] = None):
            return json.dumps(stub_verdict)

        # chat_4o_mini = _fake_chat
        try:
            res = compure_score_search(predict_str, loc)
            print_hl(f"[TEST] {name} -> {json.dumps(res, ensure_ascii=False)}")
        finally:
            chat_4o_mini = _orig_chat

    gt = {"country": "United States", "province_or_state": "California", "city": "San Francisco"}
    # Include the 'assistant' token to exercise last_predict_str extraction
    pred = "system: context...\nassistant I think it's near San Francisco, CA, USA."

    # city -> 4
    run_case(
        name="city",
        stub_verdict={"country_correct": False, "state_correct": False, "city_correct": True},
        predict_str=pred,
        loc=gt,
    )
