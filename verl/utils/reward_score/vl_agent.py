from openai import OpenAI
import requests
import random
import re
import os

from math_verify import parse, verify

openai_api_key = "EMPTY"
# openai_api_base_list = [
#     os.environ.get("LLM_AS_A_JUDGE_BASE", "http://29.225.242.90:18901/v1"),
# ]
# openai_api_base_list = ["http://29.225.242.25:18901/v1", "http://29.226.3.167:18901/v1"]
# openai_api_base_list = ["http://29.177.193.220:18901/v1", "http://29.177.112.27:18901/v1"]
openai_api_base_list = ["http://29.225.240.118:18901/v1"]
try:
    client_list = []
    for api_base in openai_api_base_list:
        client = OpenAI(
            api_key=openai_api_key,
            base_url=api_base,
        )
        client_list.append(client)
    model_name_list = []
    for client in client_list:
        response = requests.get(f"{api_base}/models")
        models = response.json()
        model_name_list.append(models['data'][0]['id'])
except:
    print(" [ERROR] Failed to initialize OpenAI client. Please check your API key and base URL.")
    client_list = []
    model_name_list = []


API_KEY = "e78f0919-4770-4dea-b0a5-c0e6b56866fb"
MODEL_NAME = "api_azure_openai_gpt-4o-mini"
MODEL_MARKER = "api_azure_openai_gpt-4o-mini"

# API_KEY = "aa57e5e3-92de-4d6c-8e60-8f57c6d4ec4a"
# MODEL_NAME = "api_openai_chatgpt-4o-latest"
# MODEL_MARKER = "api_openai_chatgpt-4o-latest"

import glob
import pandas as pd

import json

def send_request(messages):
    json_data = {
        "bid": "open_api_test",
        "server": "open_api",
        "services": [],
        "request_id": "1234",
        "session_id": "12345",  
        # "api_key": "90237a2c-0b99-4d4e-8e78-b3473ad3e13c",
        "api_key": API_KEY,
        "model_marker": MODEL_MARKER,
        "system": "", # 模型人设
        "timeout": 300, # 超时时间,单位秒
        "model_name": MODEL_NAME,
        "messages": messages,
        "params":{}
    }

    url= "http://trpc-utools-prod.turbotke.production.polaris:8009/"
    
    res = requests.post(url=url, json=json_data, proxies={"http": None, "https": None})
    # print(res)
    # print(res.json()['answer'][0]['value'])
    if res.status_code == 200 and res is not None:
        # print('input', json_data)
        if random.random() < 0.01:
            print("res.json()",res.json())
        print(res.json())
        return(res.json()['answer'][0]['value'])
    else:
        return ""

def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]

COMMON_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level reasoning problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Semantic Equivalence: Carefully examine the expression in both answers. Confirm whether the semantic meaning of student's final answer is equivalent to the reference answer, even when expressed with different wording or format.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'


    return full_prompt


def extract_answer(text):
    """
    从给定的文本中提取<answer></answer>标签内部的内容。
    
    参数:
        text (str): 包含<answer>标签的文本
        
    返回:
        str or None: 标签内部的内容，如果未找到则返回None。
    """
    # 使用非贪婪模式匹配<answer>和</answer>之间的内容
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def compute_score(predict_str: str, ground_truth: str, extra_info=None) -> float:
    
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        print(f"count_think_1 {count_think_1} != count_think_2 {count_think_2}")
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        print(f"count_vision_1 {count_vision_1} != count_vision_2 {count_vision_2}")
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        print(f"count_answer_1 {count_answer_1} != count_answer_2 {count_answer_2}")
        is_format_error = True
    
    if is_format_error:
        with open("/mnt/lzy/DeepEyes/logs/debug_crop_reflection_predstr.txt", "a") as f:
            f.write(f"{predict_str}\n\n\n\n")

    model_answer = extract_answer(predict_no_think)

    if model_answer is None:
        print(f"cannot extract <answer> from predict_no_think {predict_no_think}")
        acc_reward = 0.0
        is_format_error = True
    else:
        # if rule_math_verify(ground_truth, model_answer):
        #     acc_reward = 1.0
        try:
            if str(model_answer) == str(ground_truth):
                acc_reward = 1.0
            else:
                verify_content = generative_verify(extra_info['question'], ground_truth, model_answer)
                acc_reward = 1.0 if verify_content else 0.0
                if acc_reward == 0:
                    print(f"verify_content: {verify_content}")
                    print(f"ground_truth: {ground_truth}")
                    print(f"model_answer: {model_answer}")
                    print(f"pred_no_think: {predict_no_think}")
                # acc_reward = 0.0
        except Exception as e:
            acc_reward = 0.0
            print(f"Evaluation failed with {e}")
    # question_text = extra_info['question']
    # full_prompt = get_prompt(answer_text, ground_truth, question_text)

    # client_idx = random.randint(0, len(client_list) - 1)
    # client = client_list[client_idx]
    # model_name = model_name_list[client_idx]

    # response = ""
    # for it in range(3):
    #     try:
    #         chat_response = client.chat.completions.create(
    #             model=model_name,
    #             messages=[
    #                 {"role": "system", "content": "You are a helpful assistant."},
    #                 {"role": "user", "content": full_prompt},
    #             ],
    #             seed = random.randint(0, 1000000),
    #             temperature=0.3,
    #         )
    #         response = chat_response.choices[0].message.content.strip()
    #         break
    #     except Exception as e:
    #         print(f' [ERROR math] generative_verify error: {e}')
    #         continue
    
    # # ###: TEMPORARY DEBUG
    # # response = random.choice(['1', '0', 'Judgement: 1', 'Judgement: 0'])

    # # print(response)
    # if 'Judgement:' in response:
    #     response = response.split('Judgement:')[-1].strip()
    #     if '1' in response:
    #         acc_reward = 1.0
    #     elif '0' in response:
    #         acc_reward = 0.0
    #     else:
    #         print(f' [WARNING] resp format error {response=}')
    #         acc_reward = 0.0
    # else:
    #     if response == '1':
    #         acc_reward = 1.0
    #     elif response == '0':
    #         acc_reward = 0.0
    #     else:
    #         print(f' [WARNING] resp format error {response=}')
    #         acc_reward = 0.0

        # Penalize for model trying to predict longer answer to hack llm-as-judge
        if len(model_answer) >= 1000:
            acc_reward = 0.0
            is_format_error = True

    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    no_tool_reward = -1.0 if count_vision_1 == 0 and acc_reward < 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    # reward 1
    # return 0.8 * acc_reward + 0.2 * format_reward + 0.4 * tool_reward_base
    # reward 2
    # with open("/mnt/lzy/DeepEyes/logs/debug_crop_reward.txt", "a") as f:
    #     f.write(f"acc_reward: {acc_reward}, tool_reward: {tool_reward}, format_reward: {format_reward}\n")
    # return 1.2 * acc_reward + 0.4 * format_reward + 0.4 * tool_reward
    # reward 3
    # return 1.2 * acc_reward + 0.4 * format_reward
    # reward 4
    # from ray.util import pdb; pdb.set_trace()
    # with open("/mnt/lzy/DeepEyes/logs/debug_crop_reward3.txt", "a") as f:
    #     f.write(f"acc_reward: {acc_reward}, tool_reward: {tool_reward}, no_tool_reward: {no_tool_reward}, format_reward: {format_reward}\n")
    # score = 1.2 * acc_reward + 0.4 * format_reward + 0.8 * tool_reward + 0.4 * no_tool_reward
    # score = 1.2 * acc_reward + 0.4 * format_reward + 0.4 * tool_reward + 0.4 * no_tool_reward
    score = 1.2 * acc_reward + 0.4 * format_reward
    return_dict = {
        "score": score,
        "acc_reward": acc_reward,
        "format_reward": format_reward,
        "tool_reward": tool_reward,
        "no_tool_reward": no_tool_reward
    }
    return return_dict

    # reward 2 
    # return 1.0 * acc_reward + 0.2 * format_reward + 1.0 * tool_reward + 0.2 * tool_reward_base
    # reward 3
    # tool_reward_alpha = 1.2 if count_vision_1 > 0 else 0.0
    # return 1.0 * acc_reward * tool_reward_alpha + 0.2 * format_reward
    # reward 4
    # extra_reward = tool_reward_base * (count_vision_1 - 1) * (1 - acc_reward)
    # return  0.8 * acc_reward + 0.2 * format_reward + 0.4 * tool_reward_base  + 0.2 * extra_reward


def compute_score_crop(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    print("count_vision: ", count_vision_1, count_vision_2)
    if count_vision_1 != count_vision_2:
        is_format_error = True

    crop_tool_call_format_error = False
    count_crop = predict_str.count("Therefore, I decide to crop the image with tool_call.")
    count_nocrop = predict_str.count("Therefore, I decide not to crop the image.")
    count_tool_call = predict_str.count("<tool_call>")
    print("count_tool_call: ", count_crop, count_nocrop, count_tool_call)
    if count_crop == 0 and count_nocrop == 0:       # 都没出现
        crop_tool_call_format_error = True
    if count_crop > 0 and count_nocrop > 0:         # 都出现
        crop_tool_call_format_error = True
    if count_nocrop > 0 and count_tool_call > 0:    # 不crop
        crop_tool_call_format_error = True
    if count_crop > 0 and count_tool_call == 0:     # 要crop
        crop_tool_call_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    model_answer = extract_answer(predict_no_think)

    if model_answer is None:
        acc_reward = 0.0
        is_format_error = True
        crop_tool_call_format_error = True
    else:
        # if rule_math_verify(ground_truth, model_answer):
        #     acc_reward = 1.0
        try:
            if str(model_answer) == str(ground_truth):
                acc_reward = 1.0
            else:
                acc_reward = 1.0 if generative_verify(extra_info['question'], ground_truth, model_answer) else 0.0
                # acc_reward = 0.0
        except:
            acc_reward = 0.0

        # Penalize for model trying to predict longer answer to hack llm-as-judge
        if len(model_answer) >= 1000:
            acc_reward = 0.0
            is_format_error = True

    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    crop_tool_call_reward = -1.0 if crop_tool_call_format_error else 0.0
    # reward 1
    # return 0.8 * acc_reward + 0.2 * format_reward + 0.4 * tool_reward_base
    # reward 2
    # return 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward
    # reward 3
    return 1.6 * acc_reward + 0.4 * format_reward + 0.4 * crop_tool_call_reward


def compute_common_reasoning(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = extract_answer(predict_no_think) # predict_no_think.split("<answer>")[-1].split("</answer>")[0].strip()
    if not answer_text:
        acc_reward = 0.0
        is_format_error = True
    elif len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True
    else:
        question_text = extra_info['question']
        client_idx = random.randint(0, len(client_list) - 1)
        client = client_list[client_idx]
        model_name = model_name_list[client_idx]
        full_prompt = COMMON_VERIFY_PROMPT.format(
            query=question_text,
            gold_ans=ground_truth,
            pred_ans=answer_text,
        )

        acc_reward = 0.0
        for ix in range(5):
            # messages = [
            #         {"role": "user", "content": full_prompt},
            # ]
            # response = send_request(messages)

            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.5,
            )
            response = chat_response.choices[0].message.content.strip()
            judgement = response.split('## Equivalence Judgement')[-1].lower()

            # # TEMP DEBUG
            # judgement = random.choice(['true', 'false'])

            if 'true' in judgement and 'false' not in judgement:
                acc_reward = 1.0
                break
            elif 'false' in judgement and 'true' not in judgement:
                acc_reward = 0.0
                break
            else:
                print(f' [ERROR] judgement format invalid: {judgement}')
                continue

    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    # print(f' [DEBUG] query={extra_info["question"]}, {ground_truth=}, {answer_text=}, {acc_reward=}, {format_reward=}')
    return 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward


def rule_math_verify(ground_truth, model_answer):
    gold = parse(ground_truth)
    answer = parse(model_answer)
    return verify(gold, answer)


def generative_verify(query, ground_truth, model_answer):
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    full_prompt = COMMON_VERIFY_PROMPT.format(
        query=query,
        gold_ans=ground_truth,
        pred_ans=model_answer,
    )

    response = ""
    for it in range(5):
        try:
            # messages = [
            #     {"role": "user", "content": full_prompt},
            # ]
            # response = send_request(messages)

            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.0,
            )
            response = chat_response.choices[0].message.content.strip()

            # response = random.choice(['## Equivalence Judgement: true', '## Equivalence Judgement: false'])

            break
        except Exception as e:
            print(f' [ERROR math] generative_verify error: {e}')
            continue
    
    judgement = response.split('## Equivalence Judgement')[-1].lower()
    # print(f' [DEBUG math] query={query}, {ground_truth=}, {model_answer=}, {judgement=}')
    if 'true' in judgement and 'false' not in judgement:
        return True
    elif 'false' in judgement and 'true' not in judgement:
        return False
    else:
        print(f' [ERROR math] verify bug output: ')


def compute_score_math(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    model_answer = ""
    predict_no_think = predict_str.split('</think>')[-1].strip()
    answer_pattern = r'\\boxed{([^}]+)}'
    answer_list = re.findall(answer_pattern, predict_no_think, flags=re.DOTALL)
    if len(answer_list) == 0:
        acc_reward = 0.0
        is_format_error = True
    else:
        if len(answer_list) > 1:
            is_format_error = True

        model_answer = answer_list[-1]
        if rule_math_verify(ground_truth, model_answer):
            acc_reward = 1.0
        else:
            acc_reward = 1.0 if generative_verify(extra_info['question'], ground_truth, model_answer) else 0.0
    
    format_reward = -1.0 if is_format_error else 0.0
    print(f' [DEBUG] query={extra_info["question"]}, {ground_truth=}, {model_answer=}, {acc_reward=}, {format_reward=}')
    return 1.2 * acc_reward + 0.4 * format_reward


def compute_score_counting(predict_str: str, ground_truth: str, extra_info=None) -> float:

    # # save predict str to txt file
    # with open('/mnt/lzy/DeepEyes/predict_str.txt', 'w') as f:
    #     f.write(predict_str + '\n')

    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_tool_calling_1 = predict_str.count("<tool_call>")
    count_tool_calling_2 = predict_str.count("</tool_call>")
    if count_tool_calling_1 != count_tool_calling_2:
        is_format_error = True

    count_answer_1 = predict_str.count("<answer>")
    count_answer_2 = predict_str.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    model_answer = extract_answer(predict_no_think)
    if model_answer is None:
        acc_reward = 0.0
        is_format_error = True
    else:
        # if rule_math_verify(ground_truth, model_answer):
        #     acc_reward = 1.0
        try:
            model_answer_int = int(model_answer)
            if int(model_answer_int) == int(ground_truth):
                acc_reward = 1.0
            else:
                acc_reward = 0.0
        except:
            acc_reward = 0.0
            
    format_reward = -1.0 if is_format_error else 0.0
    print(f' [DEBUG] query={extra_info["question"]}, {ground_truth=}, {model_answer=}, {acc_reward=}, {format_reward=}')
    return 1.2 * acc_reward + 0.4 * format_reward

def compute_score_chart(predict_str: str, ground_truth: str, extra_info=None) -> float:

    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_tool_calling_1 = predict_str.count("<tool_call>")
    count_tool_calling_2 = predict_str.count("</tool_call>")
    if count_tool_calling_1 != count_tool_calling_2:
        is_format_error = True

    count_answer_1 = predict_str.count("<answer>")
    count_answer_2 = predict_str.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    model_answer = extract_answer(predict_no_think)
    if model_answer is None:
        acc_reward = 0.0
        is_format_error = True
    else:
        # if rule_math_verify(ground_truth, model_answer):
        #     acc_reward = 1.0
        try:
            if str(model_answer) == str(ground_truth):
                acc_reward = 1.0
            else:
                acc_reward = 1.0 if generative_verify(extra_info['question'], ground_truth, model_answer) else 0.0
                # acc_reward = 0.0
        except:
            acc_reward = 0.0
            
    format_reward = -1.0 if is_format_error else 0.0
    print(f' [DEBUG] query={extra_info["question"]}, {ground_truth=}, {model_answer=}, {acc_reward=}, {format_reward=}')
    return 1.2 * acc_reward + 0.4 * format_reward

import ast

def evaluate_steps(ground_truth, predicted_steps, road_pixel):
    # Convert ground_truth if it's a string
    if isinstance(ground_truth, str):
        try:
            ground_truth = ast.literal_eval(ground_truth)
        except (ValueError, SyntaxError):
            return 0.0
    
    # Convert predicted_steps if it's a string
    if isinstance(predicted_steps, str):
        try:
            predicted_steps = ast.literal_eval(predicted_steps)
        except (ValueError, SyntaxError):
            return 0.0
    
    # Check if ground_truth is a list of lists with 2 numbers each
    if not isinstance(ground_truth, list):
        return 0.0
    for point in ground_truth:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return 0.0
        if not all(isinstance(coord, (int, float)) for coord in point):
            return 0.0
    
    # Check if predicted_steps is a list
    if not isinstance(predicted_steps, list):
        return 0.0
    
    # Check each step in predicted_steps
    for step in predicted_steps:
        # Check if each step is a list or tuple of length 2 with numbers
        if not isinstance(step, (list, tuple)) or len(step) != 2:
            return 0.0
        if not all(isinstance(coord, (int, float)) for coord in step):
            return 0.0
    
    # Check duplicate steps
    seen = set()
    for step in predicted_steps:
        # 将列表转换为元组，因为列表是不可哈希的
        step_tuple = tuple(step)
        if step_tuple in seen:
            return 0.0
        seen.add(step_tuple)

    n_truth = len(ground_truth)
    n_pred = len(predicted_steps)
    correct_count = 0
    
    # Compare each step up to the minimum length
    for i in range(min(n_truth, n_pred)):
        gt_point = ground_truth[i]
        pred_point = predicted_steps[i]
        # Check if the predicted point is within the road_pixel tolerance
        if (abs(gt_point[0] - pred_point[0]) < road_pixel  and 
            abs(gt_point[1] - pred_point[1]) < road_pixel):
            correct_count += 1
    
    # Normalize by the total number of ground truth steps
    score = correct_count / n_truth if n_truth > 0 else 0.0
    return score


def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points

def is_route_clear(image, route):
    # Convert image to RGB if it's not
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    int_route = []
    for point in route:
        x, y = point
        ix = int(round(x))
        iy = int(round(y))
        int_route.append((ix, iy))
    
    visited_pixels = set()
    
    for i in range(len(int_route) - 1):
        x0, y0 = int_route[i]
        x1, y1 = int_route[i+1]
        pixels = bresenham_line(x0, y0, x1, y1)
        for (px, py) in pixels:
            if (px, py) in visited_pixels:
                continue
            visited_pixels.add((px, py))
            if px < 0 or px >= width or py < 0 or py >= height:
                continue
            pixel_value = image.getpixel((px, py))
            if pixel_value == (0, 0, 0):
                return False
    return True

def compure_score_maze(predict_str: str, ground_truth: str, extra_info=None) -> float:

    is_format_error = False
    break_wall_reward = 0.0
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_tool_calling_1 = predict_str.count("<tool_call>")
    count_tool_calling_2 = predict_str.count("</tool_call>")
    if count_tool_calling_1 != count_tool_calling_2:
        is_format_error = True

    count_answer_1 = predict_str.count("<answer>")
    count_answer_2 = predict_str.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    model_answer = extract_answer(predict_no_think)


    if model_answer is None:
        acc_reward = 0.0
        is_format_error = True
    else:
        # if rule_math_verify(ground_truth, model_answer):
        #     acc_reward = 1.0
        model_answer = model_answer.replace('The result path is ', '')
        try:
            if str(model_answer) == str(ground_truth):
                acc_reward = 1.0
            else:
                if not is_route_clear(extra_info['pil_image'], model_answer):
                    acc_reward = 0.0
                    break_wall_reward = -1.0
                else:
                    acc_reward  = evaluate_steps(ground_truth, model_answer, extra_info["road_pixel"])
        except:
            acc_reward = 0.0
            
    format_reward = -1.0 if is_format_error else 0.0
    print(f' [DEBUG] query={extra_info["question"]}, {ground_truth=}, {model_answer=}, {acc_reward=}, {format_reward=}')
    score = 1.2 * acc_reward + 0.4 * format_reward + break_wall_reward

    return_dict = {
        "score": score,
        "acc_reward": acc_reward,
        "format_reward": format_reward,
        "break_wall_reward": break_wall_reward,
    }
    return return_dict


if __name__ == '__main__':
    # predict_str = "The answer is <think> 2 + 2 = 4 </think> <answer> left </answer>"
    predict_str = "The woman is to the left of the man."
    ground_truth = "left"
    extra_info = {'answer': 'The woman is to the left of the man who is holding the camera.', 'id': 0, 'image': '/cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg', 'pred_ans': 'The woman is to the right of the man who is holding the camera.', 'question': 'Is the woman to the left or to the right of the man who is holding the camera?'}

    score = compute_score(predict_str, ground_truth, extra_info)
    score = generative_verify(extra_info['question'], ground_truth, predict_str)
    print(f"Score: {score}")
    score = compute_common_reasoning(predict_str, ground_truth, extra_info)