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

class VisualBoxTool(ToolBase):
    name = "image_box_drawing_tool"
    # user_prompt = "Here is the cropped image returned after you calling the function {}.\nIf the images provided above are sufficient to answer the user's question, please put your final answer within <answer></answer>. Otherwise you can continue to call tools within <tool_call></tool_call>."

    user_prompt = PROMPT.USER_PROMPT_BOX
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

    def execute(self, action_string: str, **kwargs) -> tuple:
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
        
            if tool_name == "image_box_drawing_tool":
                # Zoom in by cropping the image
                boxes = args['boxes']
                # image_path = args["image_path"]
                # img = Image.open(image_path)

                img = self.multi_modal_data['image'][0]
                
                # if not is_valid_point(points, img):
                #     raise ValueError(f"POINT ARGUMENTS ARE INVALID")
                marked_img = self.draw_numbered_boxes(img, boxes)
                
                # import time
                # cur_time = int(round(time.time() * 1000))
                # marked_img.save(f'/mnt/lzy/DeepEyes/examine/marked_image_{cur_time}.jpeg', format='JPEG')
                
                current_image = marked_img
                
            else:
                raise ValueError(f"Unknown tool name: {tool_name}")
            # Prepare the observation
            obs = {
                "prompt": "\n<|im_start|>user\n" + "<tool_response>" +"<image>" + self.user_prompt + "</tool_response>" + "<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": [current_image]}
            }
            reward = 0.0  # Reward for successful tool call with correct JSON
            done = False
            info = {"status": "success", "tool_used": tool_name}
            print(f'[DEBUG-image_box_drawing_tool] SUCCESS ACTION {action_string=}')
            # save to /mnt/lzy/DeepEyes/logs/success.log
            # with open('/mnt/lzy/DeepEyes/logs/success.log', 'a') as f:
            #     f.write(f'[DEBUG-image_box_drawing_tool] SUCCESS ACTION {action_string}\n')
            return obs, reward, done, info
        except Exception as e:
            # Return an error observation if something goes wrong
            print(f'[DEBUG-image_box_drawing_tool] Execute WRONG - {str(e)} {action_string=}')
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

    def draw_numbered_boxes(self, img, boxes):
        """
        在图像上绘制带数字序号的矩形框
        :param img: PIL.Image对象
        :param boxes: 边界框列表 [[x_min,y_min,x_max,y_max], ...]
        :return: 绘制后的新图像
        """
        # 创建图像副本和绘图对象
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)

        # 设置颜色
        box_color = (255, 0, 0)    # 红色边框
        text_color = (255, 255, 255)  # 白色文本
        bg_color = box_color        # 文本背景色与边框相同

        if not isinstance(boxes[0], (list, np.ndarray)):
            boxes = [boxes]

        # 遍历所有边界框
        for i, box in enumerate(boxes, start=1):
            # 验证边界框格式
            if len(box) < 4:
                continue
                
            # 提取坐标值
            coords = [float(c) for c in box[:4]]  # 确保转换为浮点数
            x_min, y_min, x_max, y_max = coords
            
            # 坐标归一化处理（如果坐标值都小于1则视为归一化坐标）
            if all(0 <= c <= 1 for c in coords):
                x_min = int(x_min * img.width)
                y_min = int(y_min * img.height)
                x_max = int(x_max * img.width)
                y_max = int(y_max * img.height)
            else:
                x_min = int(x_min)
                y_min = int(y_min)
                x_max = int(x_max)
                y_max = int(y_max)
            
            # 确保坐标有效性
            x_min, x_max = sorted([x_min, x_max])
            y_min, y_max = sorted([y_min, y_max])
            
            # 绘制矩形框
            draw.rectangle([x_min, y_min, x_max, y_max], 
                        outline=box_color, width=5)
        return img_copy
    
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