
class PROMPT():
    SYSTEM_PROMPT = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

<tools>
{"type":"function","function":{"name":"image_marking_tool","description":"Mark numbered points on an image at specified coordinates in sequential order.","parameters":{"type":"object","properties":{"points":{"type":"array","items":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2},"description":"Ordered list of coordinates where points will be marked, as [[x1,y1],[x2,y2],[x3,y3],...]"},"required":["points"]}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_marking_tool", "arguments": {"points": [[x1,y1], [x2,y2], [x3,y3]]}}  
</tool_call>
"""

    USER_PROMPT = """
\nThink first, call **image_marking_tool** if needed, then check the return image to examine whether the marks strictly one-to-one correspondent to the objects. If wrong, you can correct the mistake and call **image_marking_tool** again. If correct, you can give the answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> 
"""

    SYSTEM_PROMPT_BOX = '''You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

<tools>
{"type":"function","function":{"name":"image_box_drawing_tool","description":"Draw bounding boxes on an image at specified coordinates.","parameters":{"type":"object","properties":{"boxes":{"type":"array","items":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},"description":"Coordinates of bounding boxes where each box is defined as [x_min, y_min, x_max, y_max] representing top-left and bottom-right coordinates"}},"required":["boxes"]}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_box_drawing_tool", "arguments": {"boxes": [[10,20,100,120], [150,30,300,200]]}}  
</tool_call>
'''

    USER_PROMPT_BOX = """
\nThink first, call **image_box_drawing_tool** when bounding boxes are needed. After receiving the annotated image, verify if boxes accurately enclose target objects. If errors exist, refine coordinates and recall the tool. If correct, you can give the answer. Strictly format responses as:  
<think>Reasoning process...</think>
<tool_call>Tool JSON call</tool_call>(if tools needed)
<answer>Final solution</answer>"""

    SYSTEM_PROMPT_LINE = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

<tools>
{"type":"function","function":{"name":"image_line_drawing_tool","description":"Draw lines on a specified image by connecting coordinates in sequential order.","parameters":{"type":"object","properties":{"index":{"type":"integer","description":"Index of the image to draw on (0-based)"},"lines":{"type":"array","items":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2},"description":"Ordered list of coordinates through which lines will be drawn, as [[x1,y1],[x2,y2],[x3,y3],...]. Lines are drawn between consecutive points."}},"required":["index","lines"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_line_drawing_tool", "arguments": {"index": 0, "lines": [[x1,y1], [x2,y2], [x3,y3]]}}  
</tool_call>
"""

    USER_PROMPT_LINE = """
Think first, call **image_line_drawing_tool** if needed, then check the return image to examine whether the lines are drawn correctly to solve the problem. If wrong, you can correct the mistake and call **image_line_drawing_tool** again on a specific image index <i> as needed. If correct, you can give the answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> 
"""


    SYSTEM_PROMPT_COUNT_V2 = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Tool definition

## Image crop and zoom in tool
<tools>
{
  "type": "function",
  "function": {
    "name": "image_zoom_in_tool",
    "description": "Zoom in on a specific region of an image by cropping it based on a bounding box and applying a zoom factor",
    "parameters": {
      "properties": {
        "bbox_2d": {
          "type": "array",
          "items": {"type": "number"},
          "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."
        },
        "factor": {
          "type": "number",
          "minimum": 1.0,
          "description": "The zoom factor to apply to the cropped region. A value greater than 1.0 will enlarge the image."
        }
      },
      "required": ["bbox_2d", "factor"]
    }
  }
}
</tools>

## Image marking tool
<tools>
{
    "type": "function",
    "function": {
        "name": "image_marking_tool",
        "description": "Mark numbered points on an image at specified coordinates in sequential order.",
        "parameters": {
            "properties": {
                "index":{
                    "type":"integer",
                    "description":"Index of the image to draw on (0-based)"},
                "points": {
                    "type":"array",
                    "items": {"type": "number"},
                    "description": "Ordered list of coordinates where points will be marked, as [[x1,y1],[x2,y2],[x3,y3],...]"
                }
            },
            "required":["index", "points"]
        }
    }
}
</tools>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "factor": 2}}  
</tool_call>
<tool_call>  
{"name": "image_marking_tool", "arguments": {"index": 0, "points": [[x1,y1], [x2,y2], [x3,y3]]}}  
</tool_call>

"""

    USER_PROMPT_COUNT_V2 = """
\nThink first, call **image_zoom_in_tool** if the target objects is too small to identify or the objects are too close. Call **image_marking_tool** if needed, then check the return image to examine whether the marks strictly one-to-one correspondent to the objects. If wrong, you can correct the mistake and call **image_marking_tool** again. If correct, you can give the answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> 
"""


    SYSTEM_PROMPT_MERGE = """You are a helpful assistant.
Answer the user's question based on the image provided.
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Tool definition

## Image crop and zoom in tool
<tools>
{
"name": "image_zoom_in_tool",
"description": "Zoom in on a specific region of an image by cropping it based on a bounding box and applying a zoom factor",
"parameters": {
    "properties": {
        "bbox_2d": {
            "type": "array",
            "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. (suppose the width and height of the image are 1000)"
        },
        "factor": {
            "type": "number",
            "description": "The zoom factor to apply to the cropped region. A value greater than 1.0 will enlarge the image."
        }
    },
    "required": ["bbox_2d", "factor"]
}
}
</tools>

## Image marking tool
<tools>
{
"name": "image_marking_tool",
"description": "Mark numbered points on an image at specified coordinates in sequential order.",
"parameters": {
    "properties": {
        "index": {
            "type":"integer",
            "description": "Index of the image to draw on (0 for the original image)"},
        "points": {
            "type":"array",
            "description": "Ordered list of coordinates where points will be marked, as [[x1,y1], [x2,y2], [x3,y3],...]. (suppose the width and height of the image are 1000)"
        }
    },
    "required":["index", "points"]
}
}
</tools>

## Image line drawing tool
<tools>
{
"name":"image_line_drawing_tool",
"description":"Draw lines on a specified image by connecting coordinates in sequential order.",
"parameters":{
    "properties": {
        "index": {
            "type":"integer",
            "description":"Index of the image to draw on (0 for the original image)"
        },
        "lines": {
            "type": "array",
            "description": "Ordered list of coordinates through which lines will be drawn, as [[x1,y1], [x2,y2], [x3,y3],...]. Lines are drawn between consecutive points. (suppose the width and height of the image are 1000)"
        }
    },
    "required": ["index", "lines"]
}
}
</tools>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "factor": 2}}  
</tool_call>
<tool_call>  
{"name": "image_marking_tool", "arguments": {"index": 0, "points": [[x1,y1], [x2,y2], [x3,y3]]}}  
</tool_call>
<tool_call>  
{"name": "image_line_drawing_tool", "arguments": {"index": 0, "lines": [[x1,y1], [x2,y2], [x3,y3]]}}  
</tool_call>
"""

    USER_PROMPT_MERGE = """
\nThink first, call **image_zoom_in_tool** if the target objects is too small to identify or the objects are too close. Call **image_marking_tool** or **image_line_drawing_tool** if you need to mark points or draw lines on the image. Carefully check the return image to examine whether the tool operation shows new information, the marks or lines correctly reveals the answer to the question. If the previous operations are wrong, you can correct the mistakes and call **image_zoom_in_tool**, **image_marking_tool** or **image_line_drawing_tool** again. If correct, you can give the answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> (if reach final answer)
"""

    SYS_PROMPT_0922_AGRREGATE = """You are a helpful assistant.
Answer the user's question based on the image provided.
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Tool definition

## Image crop and zoom in tool
<tools>
{
"name": "image_zoom_in_tool",
"description": "Zoom in on a specific region of an image by cropping it based on a bounding box and applying a zoom factor",
"parameters": {
    "properties": {
        "bbox_2d": {
            "type": "array",
            "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. (suppose the width and height of the image are 1000)"
        },
        "factor": {
            "type": "number",
            "description": "The zoom factor to apply to the cropped region. A value greater than 1.0 will enlarge the image."
        }
    },
    "required": ["bbox_2d", "factor"]
}
}
</tools>

## Image marking tool
<tools>
{
"name": "image_marking_tool",
"description": "Mark numbered points on an image at specified coordinates in sequential order.",
"parameters": {
    "properties": {
        "index": {
            "type":"integer",
            "description": "Index of the image to draw on (0 for the original image)"},
        "points": {
            "type":"array",
            "description": "Ordered list of coordinates where points will be marked, as [[x1,y1], [x2,y2], [x3,y3],...]. (suppose the width and height of the image are 1000)"
        }
    },
    "required":["index", "points"]
}
}
</tools>

## Image line drawing tool
<tools>
{
"name":"image_line_drawing_tool",
"description":"Draw lines on a specified image by connecting coordinates in sequential order.",
"parameters":{
    "properties": {
        "index": {
            "type":"integer",
            "description":"Index of the image to draw on (0 for the original image)"
        },
        "lines": {
            "type": "array",
            "description": "Ordered list of coordinates through which lines will be drawn, as [[x1,y1], [x2,y2], [x3,y3],...]. Lines are drawn between consecutive points. (suppose the width and height of the image are 1000)"
        }
    },
    "required": ["index", "lines"]
}
}
</tools>

## Search web tool
<tools>
{
  "type": "function",
  "function": {
    "name": "search_web",
    "description": "Execute a web search and return normalized results containing titles, snippets, and URLs.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query string."
        }
      },
      "required": ["query"]
    }
  }
}
</tools>


**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "factor": 2}}  
</tool_call>
<tool_call>  
{"name": "image_marking_tool", "arguments": {"index": 0, "points": [[x1,y1], [x2,y2], [x3,y3]]}}  
</tool_call>
<tool_call>  
{"name": "image_line_drawing_tool", "arguments": {"index": 0, "lines": [[x1,y1], [x2,y2], [x3,y3]]}}  
</tool_call>
<tool_call>
{"name": "search_web", "arguments": {"query": "The palace museum"}}
</tool_call>"""


    SYS_PROMPT_1005 = """You are a helpful assistant.
Answer the user's question based on the image provided.
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Tool definition

## Image crop and zoom in tool
<tools>
{
"name": "image_zoom_in_tool",
"description": "Zoom in on a specific region of an image by cropping it based on a bounding box and applying a zoom factor",
"parameters": {
    "properties": {
        "bbox_2d": {
            "type": "array",
            "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. (suppose the width and height of the image are 1000)"
        },
        "factor": {
            "type": "number",
            "description": "The zoom factor to apply to the cropped region. A value greater than 1.0 will enlarge the image."
        }
    },
    "required": ["bbox_2d", "factor"]
}
}
</tools>

## Image marking tool
<tools>
{
"name": "image_marking_tool",
"description": "Mark numbered points on an image at specified coordinates in sequential order.",
"parameters": {
    "properties": {
        "index": {
            "type":"integer",
            "description": "Index of the image to draw on (0 for the original image)"},
        "points": {
            "type":"array",
            "description": "Ordered list of coordinates where points will be marked, as [[x1,y1], [x2,y2], [x3,y3],...]. (suppose the width and height of the image are 1000)"
        }
    },
    "required":["index", "points"]
}
}
</tools>

## Image line drawing tool
<tools>
{
"name":"image_line_drawing_tool",
"description":"Draw lines on a specified image by connecting coordinates in sequential order.",
"parameters":{
    "properties": {
        "index": {
            "type":"integer",
            "description":"Index of the image to draw on (0 for the original image)"
        },
        "lines": {
            "type": "array",
            "description": "Ordered list of coordinates through which lines will be drawn, as [[x1,y1], [x2,y2], [x3,y3],...]. Lines are drawn between consecutive points. (suppose the width and height of the image are 1000)"
        }
    },
    "required": ["index", "lines"]
}
}
</tools>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "factor": 2}}  
</tool_call>
<tool_call>  
{"name": "image_marking_tool", "arguments": {"index": 0, "points": [[x1,y1], [x2,y2], [x3,y3]]}}  
</tool_call>
<tool_call>  
{"name": "image_line_drawing_tool", "arguments": {"index": 0, "lines": [[x1,y1], [x2,y2], [x3,y3]]}}  
</tool_call>
"""