import json

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_POINTER_START_TOKEN = "<|pointer_start|>"
DEFAULT_POINTER_END_TOKEN = "<|pointer_end|>"
DEFAULT_POINTER_PAD_TOKEN = "<|pointer_pad|>"

# UNMASK_TOKEN_IDS = [198, 151644, 151645]

# System Message
grounding_system_message = "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."

# Chat Template
chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

assistant_template = "{% for message in messages %}{{'<|im_start|>' + message['role']}}{% if 'recipient' in message %}<|recipient|>{{ message['recipient'] }}{% endif %}{{'\n' + message['content'][0]['text']}}{% if 'end_turn' in message and message['end_turn'] %}{{'<|diff_marker|>\n'}}{% else %}{{'<|im_end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|recipient|>' }}{% endif %}"

# Special Tokens
ADDITIONAL_SPECIAL_TOKENS = [
    "<|recipient|>",
    "<|diff_marker|>",
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
]

# Action Patterns to be replaced with special tokens
ACTION_PATTENS_XY = [
    r"x=([0-9.]+), y=([0-9.]+)",
    r"from_coord=\[([0-9.]+), ([0-9.]+)\], to_coord=\[([0-9.]+), ([0-9.]+)\]",
]

until = ["<|diff_marker|>"]
