import json

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
# IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
# DEFAULT_IM_START_TOKEN = "<im_start>"
# DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_POINTER_START_TOKEN = "<|pointer_start|>"
DEFAULT_POINTER_END_TOKEN = "<|pointer_end|>"
DEFAULT_POINTER_PAD_TOKEN = "<|pointer_pad|>"

UNMASK_TOKEN_IDS = [198, 151644, 151645] # "\n", <|im_start|>, <|im_end|>

# System Message
grounding_system_message = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."

# Chat Template
chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

assistant_template = "{% for message in messages %}{{'<|im_start|>' + message['role']}}{% if 'recipient' in message %}<|recipient|>{{ message['recipient'] }}{% endif %}{{'\n' + message['content'][0]['text']}}{% if 'end_turn' in message and message['end_turn'] %}{{'<|diff_marker|>\n'}}{% else %}{{'<|im_end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|recipient|>' }}{% endif %}"

# Special Tokens
ADDITIONAL_SPECIAL_TOKENS = [
    # "<|im_start|>",
    # "<|im_end|>",
    # "<|object_ref_start|>",
    # "<|object_ref_end|>",
    # "<|box_start|>",
    # "<|box_end|>",
    # "<|quad_start|>",
    # "<|quad_end|>",
    # "<|vision_start|>",
    # "<|vision_end|>",
    # "<|vision_pad|>",
    # "<|image_pad|>",
    # "<|video_pad|>",
    "<|recipient|>",
    "<|diff_marker|>",
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
]

ACTION_PATTENS_XY = [
    r"x=([0-9.]+), y=([0-9.]+)",
    r"from_coord=\[([0-9.]+), ([0-9.]+)\], to_coord=\[([0-9.]+), ([0-9.]+)\]",

    # # pyautogui.click(x=xxx, y=xxx)
    # r"pyautogui\.click\(x=([0-9.]+), y=([0-9.]+)\)",
    # r"pyautogui\.rightClick\(x=([0-9.]+), y=([0-9.]+)\)",
    # r"pyautogui\.doubleClick\(x=([0-9.]+), y=([0-9.]+)\)",
    # # move
    # r"pyautogui\.moveTo\(x=([0-9.]+), y=([0-9.]+)\)",
    # # mobile
    # r"mobile\.swipe\(from_coord=\[([0-9.]+), ([0-9.]+)\], to_coord=\[([0-9.]+), ([0-9.]+)\]\)",
    # r"mobile\.long_press\(x=([0-9.]+), y=([0-9.]+)\)",
    # # browser
    # r"browser\.select_option\(x=([0-9.]+), y=([0-9.]+), value=['\"].+['\"]\)",
]

# Plugin Functions
select_option_func = {
    "name": "browser.select_option",
    "description": "Select an option from a dropdown menu",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the dropdown menu",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the dropdown menu",
            },
            "value": {
                "type": "string",
                "description": "The value of the option to select",
            },
        },
        "required": ["x", "y", "value"],
    },
}

swipe_func = {
    "name": "mobile.swipe",
    "description": "Swipe on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "from_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The starting coordinates of the swipe",
            },
            "to_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The ending coordinates of the swipe",
            },
        },
        "required": ["from_coord", "to_coord"],
    },
}

home_func = {"name": "mobile.home", "description": "Press the home button"}

back_func = {"name": "mobile.back", "description": "Press the back button"}

wait_func = {
    "name": "mobile.wait",
    "description": "wait for the change to happen",
    "parameters": {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
                "description": "The seconds to wait",
            },
        },
        "required": ["seconds"],
    },
}

long_press_func = {
    "name": "mobile.long_press",
    "description": "Long press on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the long press",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the long press",
            },
        },
        "required": ["x", "y"],
    },
}

open_app_func = {
    "name": "mobile.open_app",
    "description": "Open an app on the device",
    "parameters": {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "The name of the app to open",
            },
        },
        "required": ["app_name"],
    },
}

agent_system_message = f"""You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.

You have access to the following functions:
- {json.dumps(swipe_func)}
- {json.dumps(home_func)}
- {json.dumps(back_func)}
- {json.dumps(wait_func)}
- {json.dumps(long_press_func)}
- {json.dumps(open_app_func)}
"""

user_instruction = """Please generate the next move according to the ui screenshot, instruction and previous actions.

Instruction: {overall_goal}

Previous actions:
{previous_actions}
"""

until = ["<|diff_marker|>"]
