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
import os
from io import BytesIO
from math import ceil, floor
import math
import sys
from datetime import datetime
import uuid
from gpt_researcher.search_worker import run_search


# Constants
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.MULTILINE)
MAX_CROP_CALLS = 5
MAX_SEARCH_CALLS = 2
MAX_ROUNDS = 6
IMAGE_FACTOR = 28
MIN_PIXELS = 256 * 256
MAX_PIXELS = 2048 * 1024


# ----------- misc utils ----------- #

def _print_with_color(message, color):
    if color == 'red':
        print(f"\033[91m\033[1m{message}\033[0m")
    elif color == 'green':
        print(f"\033[92m\033[1m{message}\033[0m")
    elif color == 'yellow':
        print(f"\033[93m\033[1m{message}\033[0m")
    elif color == 'blue':
        print(f"\033[94m\033[1m{message}\033[0m")
    else:
        raise ValueError(f"Invalid color: {color}")


def print_error(message):
    message = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    _print_with_color(message, 'red')
    sys.stdout.flush()


def print_hl(message):
    """Highlight the message with blue color"""
    _print_with_color(message, 'blue')
    sys.stdout.flush()


# ----------- image utils ----------- #


def dump_tool_call(tool_call) -> str:
	return f"<tool_call>{json.dumps(tool_call, ensure_ascii=False, indent=2)}</tool_call>"


def get_image_resolution(image_path: str) -> Tuple[int, int]:
	"""Get original image resolution."""
	with Image.open(image_path) as img:
		return img.size


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
	"""Extract tool calls from assistant response."""
	calls: List[Dict[str, Any]] = []
	for m in TOOL_CALL_PATTERN.finditer(text or ""):
		raw = m.group(1)
		try:
			call = json.loads(raw)
			calls.append(call)
		except Exception:
			sanitized = raw.strip().strip("`")
			try:
				call = json.loads(sanitized)
				calls.append(call)
			except Exception:
				continue
	return calls

def round_by_factor(number: int, factor: int) -> int:
	return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
	return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
	return math.floor(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, 
                min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> Tuple[int, int]:
	h_bar = max(factor, round_by_factor(height, factor))
	w_bar = max(factor, round_by_factor(width, factor))
	if h_bar * w_bar > max_pixels:
		beta = math.sqrt((height * width) / max_pixels)
		h_bar = floor_by_factor(height / beta, factor)
		w_bar = floor_by_factor(width / beta, factor)
	elif h_bar * w_bar < min_pixels:
		beta = math.sqrt(min_pixels / (height * width))
		h_bar = ceil_by_factor(height * beta, factor)
		w_bar = ceil_by_factor(width * beta, factor)
	return h_bar, w_bar


def _log_kv(debug: bool, title: str, obj: Any):
	if debug:
		print_hl(title)
		try:
			print(json.dumps(obj, ensure_ascii=False))
		except Exception:
			print(str(obj))


def crop_tool_core(original_img: Image, bbox_2d: List[float], label: Optional[str] = None, debug: bool = False, abs_scaling=1., bbox_normalize=False, crop_path = None, action=None, debug_info=None) -> Dict[str, Any]:
	"""Core logic for image cropping and resizing using the original image path."""
	try:
		left, top, right, bottom = bbox_2d

		if bbox_normalize:
			w, h = original_img.size
			left_px = int(round(left / 1000.0 * w))
			top_px = int(round(top / 1000.0 * h))
			right_px = int(round(right / 1000.0 * w))
			bottom_px = int(round(bottom / 1000.0 * h))
			cropped_image = original_img.crop((left_px, top_px, right_px, bottom_px))
			w_crop = right_px - left_px
			h_crop = bottom_px - top_px
		else:
			left_px = int(left * abs_scaling)
			top_px = int(top * abs_scaling)
			right_px = int(right * abs_scaling)
			bottom_px = int(bottom * abs_scaling)
			cropped_image = original_img.crop((left_px, top_px, right_px, bottom_px))
			w_crop = right_px - left_px
			h_crop = bottom_px - top_px

		if w_crop == 0 or h_crop == 0:
			raise ValueError("Crop dimensions must be non-zero")
		aspect_ratio = max(w_crop, h_crop) / min(w_crop, h_crop)
		if aspect_ratio > 128:
			raise ValueError(f"Crop aspect ratio exceeds limit: {aspect_ratio:.2f}")
		
		new_w, new_h = smart_resize(w_crop, h_crop, factor=IMAGE_FACTOR)
		cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)

		if crop_path is None:
			tmpdir = os.path.abspath(".temp/inference")
			os.makedirs(tmpdir, exist_ok=True)
			crop_path = os.path.join(tmpdir, f"crop_{str(uuid.uuid4())}.jpg")
		cropped_image.save(crop_path, format='JPEG')

		if debug:
			_log_kv(True, "Crop saved", {
				"bbox": bbox_2d,
				"label": label,
				"abs_scaling": abs_scaling,
				"resized_wxh": [new_w, new_h],
				"crop_path": crop_path,
				"action": action,
				"debug_info": debug_info
			})

		return {
			"success": True,
			"label": label,
			"bbox": bbox_2d,
			"cropped_image": cropped_image,
			"crop_path": crop_path
		}

	except Exception as e:
		raise Exception(f"Crop tool execution failed: {str(e)}")


def search_tool_core(query: str, debug: bool = False, retriever_name: str = 'tavily') -> Dict[str, Any]:
	"""Core logic for executing a web search via the configured retriever."""
	try:
		if debug:
			_log_kv(True, "Executing web search", {"query": query, "retriever": retriever_name})
		search_results = run_search(query, retriever_name=retriever_name)
		if debug:
			_log_kv(True, "Search results (summary)", {
				"num_results": len(search_results) if isinstance(search_results, list) else "n/a"
			})
		return {
			"success": True,
			"query": query,
			"results": search_results
		}
	except Exception as e:
		raise Exception(f"Search tool execution failed: {str(e)}")
