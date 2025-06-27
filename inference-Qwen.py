import torch

from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from transformers import AutoProcessor
from transformers import AutoModelForVision2Seq
from gui_actor.constants import chat_template
from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.inference import inference

import argparse
import json

def main(args):
    # load model
    model_name = args.model_name
    data_processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = data_processor.tokenizer
    if "Qwen2.5" in model_name:
        model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        ).eval()
    elif "Qwen2" in model_name:
        model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).eval()
    else:
        raise ValueError(f"Model {model_name} not supported")

    exclude_key = "attn_scores"

    # prepare example
    dataset = load_dataset("rootsautomation/ScreenSpot")["test"]

    '''
    data structure: dict_keys(['output_text', 'n_width', 'n_height', 'attn_scores', 'topk_points', 'topk_values', 'topk_points_all'])
    '''
    with open(args.save_path, "w", encoding="utf-8") as f:
        for batch in dataset:
            # define conversation template
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.",
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": batch["image"], # PIL.Image.Image or str to path
                            # "image_url": "https://xxxxx.png" or "https://xxxxx.jpg" or "file://xxxxx.png" or "data:image/png;base64,xxxxxxxx", will be split by "base64,"
                        },
                        {
                            "type": "text",
                            "text": batch["instruction"]
                        },
                    ],
                },
            ]

            print(f"Intruction: {batch['instruction']}")
            print(f"ground-truth action region (x1, y1, x2, y2): {[round(i, 2) for i in batch['bbox']]}") # bbox GT

            # inference
            with torch.no_grad():
                pred = inference(conversation, model, tokenizer, data_processor, use_placeholder=True, topk=3)
            
            filtered = {
                'instruction': batch['instruction'],
                'bbox': [round(i, 2) for i in batch['bbox']]
            }
            filtered.update({k: v for k, v in pred.items() if k != exclude_key})
            
            # round the values
            f.write(json.dumps(filtered, ensure_ascii=False) + "\n")
            # immediately record the result to jsonl for each batch
            f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/GUI-Actor-7B-Qwen2.5-VL")
    parser.add_argument("--device_map", type=str, default="cuda:0")
    parser.add_argument("--save_path", type=str, default="./results/Qwen2.5-VL-7B-example.jsonl")

    args = parser.parse_args()
    main(args)