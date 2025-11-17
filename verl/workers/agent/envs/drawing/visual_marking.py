import numpy as np
import copy
from verl.workers.agent.tool_envs import ToolBase
from typing import Optional, List, Dict, Any
from PIL import Image
import re
import json
from verl.workers.agent.envs.drawing.prompt import PROMPT
from math import ceil, floor
from PIL import Image, ImageDraw, ImageFont
import math

def compute_crop_scaled_size(resize_h, resize_w, crop_h, crop_w, max_len=32768, add_len=4000):
    """
    计算crop图像的缩放因子，保证总的prompt length不超过max_len。
    保持长宽比，不放大，只缩小。
    """
    A = (resize_h * resize_w) / (28 * 28) + add_len
    B = (crop_h * crop_w) / (28 * 28)

    if B <= 0:
        return 1.0  # 没有crop图，scale=1

    max_sqr = (max_len - A) / B
    if max_sqr <= 0:
        return 0.0  # 已经超限，无法再加crop图

    s = math.sqrt(max_sqr)
    s = min(s, 1.0)

    scaled_h = int(crop_h * s)
    scaled_w = int(crop_w * s)

    # 向下取整到 28 的倍数
    scaled_h = max(28, (scaled_h // 28) * 28)
    scaled_w = max(28, (scaled_w // 28) * 28)

    return scaled_h, scaled_w

class VisualMarkingTool(ToolBase):
    name = "image_marking_tool"
    user_prompt = PROMPT.USER_PROMPT_COUNT_V2
    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current image being processed


    def extract_answer(self, action_string: str) -> Dict[str, any]:
        answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer[-1] if answer else None
        
    def extract_action(self, action_string: str) -> Dict[str, Any]:
        """
        Extracts the tool call from the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            A dictionary with the tool name and arguments.
            
        Raises:
            ValueError: If no tool call is found or JSON is invalid.
        """
        tool_call_match = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return tool_call_match[-1] if tool_call_match else None

    def execute(self, action_string: str, vllm_input_list, **kwargs) -> tuple:
        """
        Execute the tool functionality based on the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            observation: The structured observation with the processed image.
            reward: 0.1 if tool call is successful with correct JSON format, 0 otherwise.
            done: Whether the episode is terminated.
            info: Additional info.
        """
        
        answer = self.extract_answer(action_string)
        if answer:
            return "", 0.0, True, {}
        action = self.extract_action(action_string)
        if not action:
            return "", 0.0, True, {}
        try:
            tool_call = json.loads(action.strip())  # 或使用 literal_eval
        except Exception as e:
            error_msg = f"Invalid tool call format: {action.strip()}. Error: {e}"
            obs = "\n<|im_start|>user\n" + f"Error: {str(error_msg)}" + "<|im_end|>\n<|im_start|>assistant\n"
            info = {"error": str(e), "status": "failed"}
            return obs, 0.0, False, {}
        try:

            tool_name = tool_call["name"]
            args = tool_call["arguments"]
        
            if tool_name == "image_marking_tool":
                # Zoom in by cropping the image
                points = args["points"]
                if 'index' in args.keys():
                    image_index = args['index']
                else:
                    image_index = 0
                # image_path = args["image_path"]
                # img = Image.open(image_path)
                img = vllm_input_list['multi_modal_data']['image'][image_index]
                # if not is_valid_point(points, img):
                #     raise ValueError(f"POINT ARGUMENTS ARE INVALID")
                marked_img = self.draw_numbered_points(img, points)
                marked_img_idx = len(vllm_input_list['multi_modal_data']['image'])
                current_image = marked_img
                # import time
                # cur_time = int(round(time.time() * 1000))
                # marked_img.save(f'/mnt/lzy/DeepEyes/logs/debug_images/marked_image_{cur_time}.jpeg', format='JPEG')
                current_image = marked_img
            elif tool_name == "image_zoom_in_tool":
                bbox_2d = args["bbox_2d"]
                bbox = self.maybe_resize_bbox(*bbox_2d)
                if not bbox:
                    raise ValueError(f"ZOOM IN ARGUMENTS ARE INVALID")
                
                factor = args["factor"]
                img = vllm_input_list['multi_modal_data']['image'][0]
                cropped_img = img.crop(bbox)

                w, h = img.size
                crop_w, crop_h = cropped_img.size
                new_w = int(crop_w * factor)
                new_h = int(crop_h * factor)
                new_safe_h, new_safe_w = compute_crop_scaled_size(h, w, new_h, new_w)
                zoomed_img = cropped_img.resize((new_safe_w, new_safe_h), Image.BICUBIC)
                current_image = zoomed_img
                marked_img_idx = len(vllm_input_list['multi_modal_data']['image'])

            else:
                raise ValueError(f"Unknown tool name: {tool_name}")
            # Prepare the observation
            obs = {
                "prompt": "\n<|im_start|>user\n" + "<tool_response>" + f'<index {marked_img_idx}>' + "<image>" + "</tool_response>" + "<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": [current_image]}
            }
            reward = 0.0  # Reward for successful tool call with correct JSON
            done = False
            info = {"status": "success", "tool_used": tool_name}
            print(f'[DEBUG-image_marking_tool] SUCCESS ACTION {action_string=}')
            # save to /mnt/lzy/DeepEyes/logs/success.log
            # with open('/mnt/lzy/DeepEyes/logs/success.log', 'a') as f:
            #     f.write(f'[DEBUG-image_marking_tool] SUCCESS ACTION {action_string}\n')
            return obs, reward, done, info
        except Exception as e:
            # Return an error observation if something goes wrong
            print(f'[DEBUG-image_marking_tool] Execute WRONG - {str(e)} {action_string=}')
            with open('/mnt/lzy/DeepEyes/logs/error.log', 'a') as f:
                f.write(f'[DEBUG-image_marking_tool] Execute WRONG - {str(e)} {action_string}\n')
            obs = "\n<|im_start|>user\n" + f"Error: {str(e)}" + "<|im_end|>\n<|im_start|>assistant\n"
            reward = 0.0  # No reward for failed execution
            done = False
            info = {"error": str(e), "status": "failed"}
            return obs, reward, done, info

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'
        
        self.height = self.multi_modal_data['image'][0].height
        self.width = self.multi_modal_data['image'][0].width

    def maybe_resize_bbox(self, left, top, right, bottom):
        left = max(0, left)
        top = max(0, top)
        right = min(self.width, right)
        bottom = min(self.height, bottom)
        if not self.validate_bbox(left, top, right, bottom):
            return None

        height = bottom - top
        width = right - left
        if height < 28 or width < 28:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 28 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)
            if not self.validate_bbox(new_left, new_top, new_right, new_bottom):
                return None
            return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]

    def draw_numbered_points(self, img, points):
        """
        在图像上绘制带数字序号的点
        :param img: PIL.Image对象
        :param points: 点坐标列表 [[x1,y1], [x2,y2], ...]
        :return: 绘制后的新图像
        """
        # 创建图像副本和绘图对象
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # 尝试加载字体（支持跨平台）
        try:
            # 尝试常见字体
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 20)
            except:
                # 使用默认字体（大小不可调）
                font = ImageFont.load_default()
        
        # 设置颜色
        point_color = (255, 0, 0)  # 红色圆点
        text_color = (255, 255, 255)  # 白色文本
        
        # 遍历所有点
        for i, point in enumerate(points, start=1):
            # 验证点坐标
            if len(point) < 2:
                continue
                
            x, y = point[:2]  # 取前两个值作为坐标

            if x < 1 and y < 1:
                x = int(x * img.width)
                y = int(y * img.height)
            else:
                x = int(x)
                y = int(y)

            # 检查坐标是否在图像范围内
            if 0 <= x < img.width and 0 <= y < img.height:
                # 绘制红色圆点
                point_radius = 5
                draw.ellipse(
                    [x - point_radius, y - point_radius,
                    x + point_radius, y + point_radius],
                    fill=point_color
                )
                
                # 绘制带背景的数字标签
                text = str(i)
                text_position = (x + 8, y - 10)  # 点在数字左侧
                
                # 计算文本尺寸（不同字体处理方法不同）
                if hasattr(font, 'getbbox'):  # Pillow 9.2.0+
                    bbox = font.getbbox(text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:  # 旧版本Pillow
                    text_width, text_height = font.getsize(text)
                
                # 绘制文本背景
                bg_margin = 2
                draw.rectangle(
                    [text_position[0] - bg_margin, 
                    text_position[1] - bg_margin,
                    text_position[0] + text_width + bg_margin, 
                    text_position[1] + text_height + bg_margin],
                    fill=point_color
                )
                
                # 绘制文本
                draw.text(text_position, text, fill=text_color, font=font)
        
        return img_copy

if __name__ == "__main__":
    # Example usage (for testing)
    tool = VisualMarkingTool("visual_toolbox", "Tool for image processing", {})
    
    # Test zoom in tool (should return reward=0.1)
    zoom_in_action = """
    <tool_call>
    {"name": "image_marking_tool", "arguments": {"image_path": "/mnt/lzy/DeepEyes/1625003769.8128223.jpeg", "points": [
    [30, 12],
    [36, 96],
    [38, 96],
    [41, 96],
    [43, 97],
    [46, 96],
    [46, 85],
    [43, 84],
    [41, 85],
    [38, 84],
    [34, 82],[31, 84]]}}
    </tool_call>
    """
    import pdb; pdb.set_trace()
    obs, reward, done, info = tool.execute(zoom_in_action)
    print(f"Zoom in result - Reward: {reward}, Info: {info}")
    

    # Test zoom in tool (should return reward=0.1)
    zoom_in_action = """
    <tool_call>
    {"name": "image_marking_tool", "arguments": {"image_path": "/mnt/lzy/DeepEyes/1625003769.8128223.jpeg", "points": [
    [30.1234, 12.23],
    [36.3, 96.0],
    [38.532, 96.33],
    [41.02, 96.63],
    [43.86, 97.59],
    [46.01, 96.0],
    [46, 85],
    [43, 84],
    [41, 85],
    [38, 84],
    [34, 82],[31, 84]]}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(zoom_in_action)
    print(f"Zoom in result - Reward: {reward}, Info: {info}")

    # Test invalid JSON (should return reward=0.0)
    invalid_action = """
    <tool_call>
    {"name": "image_marking_tool", "arguments": {"image_path": "/mnt/lzy/DeepEyes/625003769.8128223.jpeg", "angle": 90}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(invalid_action)
    print(f"Invalid JSON result - Reward: {reward}, Info: {info}")
    
    # Test unknown tool (should return reward=0.0)
    unknown_tool_action = """
    <tool_call>
    {"name": "unknown_tool", "arguments": {"param": "value"}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(unknown_tool_action)
    print(f"Unknown tool result - Reward: {reward}, Info: {info}")