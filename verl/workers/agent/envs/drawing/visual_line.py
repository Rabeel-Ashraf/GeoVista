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

class VisualLineTool(ToolBase):
    name = "image_line_drawing_tool"
    # user_prompt = "Here is the cropped image returned after you calling the function {}.\nIf the images provided above are sufficient to answer the user's question, please put your final answer within <answer></answer>. Otherwise you can continue to call tools within <tool_call></tool_call>."

    user_prompt = PROMPT.USER_PROMPT_LINE
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
        
            if tool_name == "image_line_drawing_tool":
                image_index =args['index']
                image_lines = args['lines']
                # img = self.multi_modal_data['image'][image_index]
                img = vllm_input_list['multi_modal_data']['image'][image_index]
                marked_img = img.copy()
                for point_idx in range(len(image_lines) - 1):
                    marked_img = draw_line(marked_img, image_lines[point_idx], image_lines[point_idx + 1])
                marked_img_idx = len(vllm_input_list['multi_modal_data']['image'])
                current_image = marked_img
                
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
            print(f'[DEBUG-image_line_drawing_tool] SUCCESS ACTION {action_string=}')
            # save to /mnt/lzy/DeepEyes/logs/success.log
            # with open('/mnt/lzy/DeepEyes/logs/success.log', 'a') as f:
            #     f.write(f'[DEBUG-image_box_drawing_tool] SUCCESS ACTION {action_string}\n')
            return obs, reward, done, info
        except Exception as e:
            # Return an error observation if something goes wrong
            # print("image list: ", self.multi_modal_data['image'])
            # print("chat_history: ", self.chatml_history)
            # print(vllm_input_list)
            print(f'[DEBUG-image_line_drawing_tool] Execute WRONG - {str(e)} {action_string=}')
            # with open('/mnt/lzy/DeepEyes/logs/error.log', 'a') as f:
            #     f.write(f'[DEBUG-image_box_drawing_tool] Execute WRONG - {str(e)} {action_string}\n')
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


def draw_line(img, point_start, point_end):
    """
    在图像上绘制带数字序号的点
    :param img: PIL.Image对象
    :param points: 点坐标列表 [[x1,y1], [x2,y2], ...]
    :return: 绘制后的新图像
    """
    # 创建图像副本和绘图对象
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # 设置颜色
    point_color = (255, 0, 0)  # 红色圆点
    bg_margin = 3

    x1, y1 = point_start[0], point_start[1]
    x2, y2 = point_end[0], point_end[1]

    draw.rectangle(
                [min(x1, x2) - bg_margin, 
                min(y1, y2) - bg_margin,
                max(x1, x2)  + bg_margin, 
                max(y1, y2) + bg_margin],
            fill=point_color)
    return img_copy.convert('RGB')  # 转换为RGB模式以避免透明度问题

    
if __name__ == "__main__":
    # Example usage (for testing)
    tool = VisualBoxTool("visual_toolbox", "Tool for image processing", {})
    
    # Test zoom in tool (should return reward=0.1)
    zoom_in_action = """
    <tool_call>
    {"name": "image_box_drawing_tool", "arguments": {"image_path": "/mnt/lzy/DeepEyes/tools/chart/chart3.png", "boxes": [[372,284,596,590]]}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(zoom_in_action)
    print(f"Zoom in result - Reward: {reward}, Info: {info}")
    import pdb; pdb.set_trace()
    
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