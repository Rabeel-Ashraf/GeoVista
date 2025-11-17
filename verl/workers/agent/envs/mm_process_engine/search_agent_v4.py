import re
import random
import requests
import numpy as np
import requests
import base64
import json

from typing import Any, Dict, List, Optional, Tuple
from time import sleep
from PIL import Image
from io import BytesIO
import os
from math import ceil, floor
import math
import uuid
import traceback

from verl.workers.agent.tool_envs import ToolBase, extract_tool_call_contents

try:
	from .search_utils_v1 import smart_resize, extract_tool_calls, MAX_CROP_CALLS, MAX_SEARCH_CALLS, crop_tool_core, search_tool_core, MAX_ROUNDS, dump_tool_call
except:
	print("cannot import search_utils_v1")

def format_usr_message(msg: str)->str:
	chat_template = """<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""
	return chat_template.format(msg)


class SearchAgentEnvV4(ToolBase):
	name = "search_agent_v4"
	temp_dir = "/mnt/private/agent_workspace/hunyuan-o3/.temp/outputs/inference"
	
	def __init__(self, _name, _desc, _params, **kwargs):
		self.chatml_history = []
		self.multi_modal_data = None
		self.reset_worker()
		self.adaptive_scaling = 1.0
		self.resized_height: Optional[int] = None
		self.resized_width: Optional[int] = None

		super().__init__(name=self.name)
	

	def reset_worker(self):
		self.crop_call_count = 0
		self.search_call_count = 0
		self.round_count = 0
	

	def execute_crop_tool(self, bbox_2d: List[float], label: Optional[str] = None, debug: bool = False, abs_scaling=1., action=None) -> Dict[str, Any]:
		"""Execute image crop tool call."""
		if self.crop_call_count >= MAX_CROP_CALLS:
			raise Exception(f"Maximum crop tool calls ({MAX_CROP_CALLS}) exceeded")
		
		debug_info = {'height': self.height, 'width': self.width, 'compressed_height': self.resized_height, 'compressed_width': self.resized_width}
		
		try:
			os.makedirs(self.temp_dir, exist_ok=True)
			crop_filename = f"crop_{self.crop_call_count}_{str(uuid.uuid4())}.jpg"
			crop_path = os.path.join(self.temp_dir, crop_filename)

			result = crop_tool_core(
				original_img=self.multi_modal_data['image'][0],
				bbox_2d=bbox_2d,
				label=label,
				debug=debug,
				abs_scaling=abs_scaling,
				crop_path=crop_path,
				action=action,
				debug_info=debug_info
			)

			self.crop_call_count += 1
			return result
			
		except Exception as e:
			raise Exception(f"Crop tool execution failed: {str(e)}")
	
	def execute_search_tool(self, query: str, debug: bool = False) -> Dict[str, Any]:
		"""Execute web search tool call."""
		if self.search_call_count >= MAX_SEARCH_CALLS:
			raise Exception(f"Maximum search tool calls ({MAX_SEARCH_CALLS}) exceeded")
		
		try:
			result = search_tool_core(query=query, debug=debug, retriever_name='tavily')
			self.search_call_count += 1
			return result
			
		except Exception as e:
			raise Exception(f"Search tool execution failed: {str(e)}")
    

	def should_prompt_search(self) -> bool:
		"""Check if we should prompt for search in the next round."""
		return (self.round_count == MAX_ROUNDS - 2 and 
				self.search_call_count == 0)
	

	def should_force_final_answer(self) -> bool:
		"""Check if we should force final answer."""
		return self.round_count == MAX_ROUNDS - 1
    
	def increment_round(self):
		"""Increment round counter."""
		self.round_count += 1
	
	def wrap_tool_response(self, str_result):
		return f'<tool_response>{str_result}</tool_response>'

	def execute(self, action_string, vllm_input_list, **kwargs):
		# obs_dict, step_reward, done, info
		debug = kwargs["debug"] if "debug" in kwargs else False
		extra_info = kwargs["extra_info"] if "extra_info" in kwargs else None 
		print(extra_info)

		# TODO: remove this
		debug = True
		
		info_dict = {
			"tool_success": 0,
			"tool_fail": 0,
			"search_web": 0,
		}

		crop_img = None
		has_answer = False
		if self.round_count < MAX_ROUNDS:
			tool_calls = extract_tool_calls(action_string)
			# tool calls
			if tool_calls:
				tool_calls = tool_calls[:1]
				for call in tool_calls:
					try:
						name = call.get("name")
						arguments = call.get("arguments", {})
						if name == "image_zoom_in_tool":
							bbox_2d = arguments.get("bbox_2d")
							label = arguments.get("label")
							result = self.execute_crop_tool(
								bbox_2d,
								label,
								debug=debug,
								abs_scaling=self.adaptive_scaling,
								action=action_string
							)
							usr_obs = f"For the image, You have zoomed in on the following area: {dump_tool_call(call)}, and the cropped image is as follows: <image>"
							crop_img = result["cropped_image"]
							usr_obs = self.wrap_tool_response(usr_obs)

						elif name == "search_web":
							query = arguments.get("query")
							result = self.execute_search_tool(query, debug=debug)
							search_result_json = json.dumps({
								"name": "search_web",
								"query": result["query"],
								"result": result["results"]
							}, ensure_ascii=False, indent=2)
							usr_obs = f"For {dump_tool_call(call)}, the search results are as follows:\n{search_result_json}"
							usr_obs = self.wrap_tool_response(usr_obs)
						
						else:
							raise ValueError(f"Unknown tool call: `{name}`, please check the tool call and try again.")
						
						if debug:
							print(f'[DEBUG-search_agent_v1] SUCCESS ACTION {action_string=}')
						info_dict["tool_success"] += 1

					except Exception as e:
						# raise e
						if 'TAVILY_API_KEY' in str(e):
							raise ValueError(f"TAVILY_API_KEY is not set, please set the TAVILY_API_KEY in the environment variables.")
						tb_lines = traceback.format_exc().strip().split('\n')
						key_info = '\n'.join(tb_lines[-6:]) if len(tb_lines) > 6 else traceback.format_exc()
						usr_obs = f"Tool call error: {str(e)}\n\nError details:\n{key_info}"

						if debug:
							print(f'[DEBUG-search_agent_v1] ERROR ACTION {action_string=} {usr_obs}')
						info_dict["tool_fail"] += 1
				
				if self.should_prompt_search():
					usr_obs += "You should consider using web search tool to find more information about the location."
				elif self.should_force_final_answer():
					usr_obs += "Now you must try to identify the place where the original image is located, without more tool uses."
			
			# no tool calls
			else:
				usr_obs = ""
				has_answer = True
				if debug:
					print(f'[DEBUG-search_agent_v1] FINISH ACTION {action_string=}')
		else:
			usr_obs = "You have reached the maximum number of rounds. Please provide your final answer."
			if debug:
				print(f'[DEBUG-search_agent_v1] EXCEED ACTION {action_string=}')
        
		self.increment_round()

		if crop_img is not None:
			obs_dict = {"prompt": format_usr_message(usr_obs), "multi_modal_data": {"image": [crop_img]}}
		else:
			obs_dict = format_usr_message(usr_obs)
		return obs_dict, 0.0, has_answer, info_dict
		

	def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
		self.chatml_history = raw_prompt
		self.multi_modal_data = origin_multi_modal_data

		assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
		assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'
		self.height = self.multi_modal_data['image'][0].height
		self.width = self.multi_modal_data['image'][0].width

		resized_h, resized_w = smart_resize(self.height, self.width)
		self.resized_height = resized_h
		self.resized_width = resized_w
		self.adaptive_scaling = (self.width / resized_w) if resized_w else 1.0
		# self.adaptive_scaling = 1.0
		
		self.reset_worker()


if __name__ == '__main__':
	tool = SearchAgentEnvV4(_name=None, _desc=None, _params=None)
	img_path = "/mnt/private/agent_workspace/hunyuan-o3/.temp/datasets/google_javascript_maps/00011929-777f-4448-a931-21257ea1fc14/panorama-00011929-777f-4448-a931-21257ea1fc14.png"

	if not os.path.exists(img_path):
		raise FileNotFoundError(f"Image not found: {img_path}")

	img = Image.open(img_path).convert("RGB")

	origin_multi_modal_data = {"image": [img]}
	raw_prompt = []

	tool.reset(raw_prompt=raw_prompt, multi_modal_data=origin_multi_modal_data, origin_multi_modal_data=origin_multi_modal_data)

	# Test 1: zoom-in crop on center region
	w, h = img.width, img.height
	x1, y1 = int(0.30 * w), int(0.30 * h)
	x2, y2 = int(0.70 * w), int(0.70 * h)

	action_text_1 = f"""<think>Zoom into the center region to inspect details.</think>
<tool_call>
{{"name":"image_zoom_in_tool","arguments":{{"bbox_2d":[{x1},{y1},{x2},{y2}],"label":"center"}}}}
</tool_call>"""

	observation, reward, done, info = tool.execute(action_string=action_text_1)
	print("Test1 (crop) observation:")
	print(observation)

	# Save/print cropped image info if available
	if isinstance(observation, dict):
		mm = observation.get("multi_modal_data", {})
		imgs = mm.get("image", []) if isinstance(mm, dict) else []
		if imgs:
			crop_out = imgs[0]
			try:
				if isinstance(crop_out, Image.Image):
					save_path = os.path.join(tool.temp_dir, "test_crop.jpg")
					crop_out.save(save_path)
					print(f"Saved cropped image to: {save_path}")
				elif isinstance(crop_out, str):
					print(f"Cropped image path: {crop_out}")
			except Exception as e:
				print(f"Failed to handle cropped image: {e}")

	# Test 2: web search (optional; may require API key)
	try:
		action_text_2 = """<think>Need to search the web for more context.</think>
<tool_call>
{"name":"search_web","arguments":{"query":"famous iron lattice tower in Paris"}}
</tool_call>"""
		observation2, reward2, done2, info2 = tool.execute(action_string=action_text_2)
		print("Test2 (search) observation:")
		out_str = str(observation2)
		print(out_str if len(out_str) < 2000 else out_str[:2000] + " ...")
	except Exception as e:
		print(f"Search test skipped/failed: {e}")

	# Test 3: no tool call -> final answer attempt
	action_text_3 = "Providing a final answer without any tool calls."
	observation3, reward3, done3, info3 = tool.execute(action_string=action_text_3)
	print("Test3 (no tool) observation:")
	print(observation3)
