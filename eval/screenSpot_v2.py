import torch
import os
import json
import argparse

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor

from gui_actor.constants import chat_template
from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.inference import inference, ForceFollowTokensLogitsProcessor
from gui_actor.utils import do_boxes_overlap
from gui_actor.constants import DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_END_TOKEN

IMAGE_PATCH_SIZE =14

def normalize_bbox(bbox_x1y1x2y2, img_width, img_height):
    # if bbox_x1y1x2y2 is not normalized to [0, 1], normalize it
    x1, y1, x2, y2 = bbox_x1y1x2y2
    if (0 <= x1 <= 1) and (0 <= y1 <= 1) and (0 <= x2 <= 1) and (0 <= y2 <= 1):
        return bbox_x1y1x2y2
    else:
        x1 = x1 / img_width
        y1 = y1 / img_height
        x2 = x2 / img_width
        y2 = y2 / img_height
        return x1, y1, x2, y2

def evaluate(model_name_or_path, model_type, use_placeholder, topk):
    # initialize model
    data_processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = data_processor.tokenizer
    for k, v in tokenizer.added_tokens_encoder.items():
        print(v, k)

    if model_type == "qwen2vl":
        print(f"Loading model with Qwen2-VL backbone from {model_name_or_path}")
        model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2"
        ).eval()
        grounding_system_message = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."
    elif model_type == "qwen25vl":
        print(f"Loading model with Qwen2.5-VL backbone from {model_name_or_path}")
        model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2"
        ).eval()
        grounding_system_message = "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    print(f"Loaded model from {model_name_or_path}")

    logits_processor_pointer = ForceFollowTokensLogitsProcessor(
        token_a_id=tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0],
        forced_sequence=[
            tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
        ]
    )

    dataset = load_dataset("HongxinLi/ScreenSpot_v2")["test"]
    domain_dict = {
            "windows": "desktop",
            "macos": "desktop",
            "ios": "mobile",
            "android": "mobile",
            "tool": "web",
            "shop": "web",
            "gitlab": "web",
            "forum": "web"
        }

    results = []
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        ele = {
            "file_name": example["file_name"],
            "data_type": example["data_type"],
            "domain": domain_dict[example["data_source"]],
            "instruction": example["instruction"],
            "img_size": example["image"].size,
            "bbox_x1y1x2y2": normalize_bbox(example["bbox"], example["image"].size[0], example["image"].size[1]),
            "hit_top1": 0,
            "overlap_top1": 0,
            "hit_topk": 0,
            "overlap_topk": 0,
        }

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": grounding_system_message,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": example["image"], # PIL.Image.Image or str to path
                        # "image_url": "https://xxxxx.png" or "https://xxxxx.jpg" or "file://xxxxx.png" or "data:image/png;base64,xxxxxxxx", will be split by "base64,"
                    },
                    {
                        "type": "text",
                        "text": example["instruction"]
                    },
                ],
            },
        ]

        pred = inference(conversation, model, tokenizer, data_processor, logits_processor=logits_processor_pointer, use_placeholder=use_placeholder, topk=3)
        topk_points = pred["topk_points"]
        gt_bbox = ele["bbox_x1y1x2y2"]

        # compute the metrics
        px, py = topk_points[0]
        x1, y1, x2, y2 = gt_bbox

        if (x1 <= px <= x2) and (y1 <= py <= y2):
            ele["hit_top1"] = 1
            ele["hit_topk"] = 1

        w, h = example["image"].size
        pred_bbox = [px - IMAGE_PATCH_SIZE / w, py - IMAGE_PATCH_SIZE / h, px + IMAGE_PATCH_SIZE / w, py + IMAGE_PATCH_SIZE / h]
        if do_boxes_overlap(pred_bbox, gt_bbox):
            ele["overlap_top1"] = 1
            ele["overlap_topk"] = 1

        for px, py in topk_points[1:]:
            if (x1 <= px <= x2) and (y1 <= py <= y2):
                ele["hit_topk"] = 1
            pred_bbox = [px - IMAGE_PATCH_SIZE / w, py - IMAGE_PATCH_SIZE / h, px + IMAGE_PATCH_SIZE / w, py + IMAGE_PATCH_SIZE / h]
            if do_boxes_overlap(pred_bbox, gt_bbox):
                ele["overlap_topk"] = 1

        results.append(ele)
    
    return results


def get_metric(list_of_examples, 
               domains=["mobile", "desktop", "web"],
               data_types=["text", "icon"]):
    """
    Computes metrics over a list of examples and prints/plots a table.
    
    Each element in list_of_examples is a dict containing:
        - "domain": Domain name (e.g., "web", "mobile", "desktop")
        - "data_type": Data type (e.g., "text", "icon")
        - "hit_top1", "overlap_top1", "hit_topk", "overlap_topk": binary (0 or 1)
    
    The final table has columns for each domain broken down by UI type (plus a domain-average)
    and overall columns ("All-text", "All-icon", "All-average").
    
    The rows of the table are:
        - hit_top1
        - overlap_top1
        - hit_topk
        - overlap_topk
    """
    
    # List of metric keys to compute.
    metrics = ["hit_top1", "overlap_top1", "hit_topk", "overlap_topk"]

    # Helper function to compute the mean of a given key from a list of examples.
    def compute_mean(examples, key):
        if not examples:
            return None
        return sum(example.get(key, 0) for example in examples) / len(examples)

    # Prepare results dictionary: structure {metric: {column_name: value}}.
    results = {metric: {} for metric in metrics}
    
    # Compute metrics for each group broken down by UI type.
    for domain in domains:
        # Filter examples for the current group.
        domain_examples = [ex for ex in list_of_examples if ex.get("domain") == domain]
        for data_type in data_types:
            # Filter further for the specific UI type.
            domain_data_type_examples = [ex for ex in domain_examples if ex.get("data_type") == data_type]
            col_name = f"{domain}-{data_type}"
            for metric in metrics:
                results[metric][col_name] = compute_mean(domain_data_type_examples, metric)
        
        # Compute domain-average (all UI types for this domain).
        col_name_avg = f"{domain}-avg"
        for metric in metrics:
            results[metric][col_name_avg] = compute_mean(domain_examples, metric)

    # Compute overall metrics for each UI type across all domains.
    for data_type in data_types:
        data_type_examples = [ex for ex in list_of_examples if ex.get("data_type") == data_type]
        col_name = f"All-{data_type}"
        for metric in metrics:
            results[metric][col_name] = compute_mean(data_type_examples, metric)
    
    # Compute overall average across all examples.
    overall_key = "All-avg"
    for metric in metrics:
        results[metric][overall_key] = compute_mean(list_of_examples, metric)
    
    # Define the order of columns.
    columns_order = []
    for domain in domains:
        for data_type in data_types:
            columns_order.append(f"{domain}-{data_type}")
        columns_order.append(f"{domain}-avg")
    for data_type in data_types:
        columns_order.append(f"All-{data_type}")
    columns_order.append("All-avg")
    
    # ------------- Print Table to Console -------------
    # Prepare header row.
    header = [""] + columns_order
    # Calculate column widths for console printing.
    col_widths = [max(len(col), 12) for col in header]
    
    def format_cell(cell):
        if isinstance(cell, float):
            return f"{cell*100:.2f}"
        elif cell is None:
            return "N/A"
        return str(cell)
    
    # Print header.
    header_line = " | ".join(word.ljust(width) for word, width in zip(header, col_widths))
    separator_line = "-+-".join("-" * width for width in col_widths)
    print(header_line)
    print(separator_line)
    
    for metric in metrics:
        row = [metric]
        for col in columns_order:
            val = results[metric].get(col)
            row.append(format_cell(val))
        row_line = " | ".join(word.ljust(width) for word, width in zip(row, col_widths))
        print(row_line)
    
    # ------------- Print Tab-delimited Version (for Excel Copy-Paste) -------------
    metric_info = "Tab-delimited Table for Excel:\n"
    # Header row.
    header_tab = "\t".join([""] + columns_order)
    metric_info += (header_tab + "\n")
    # Each row.
    for metric in metrics:
        row = [metric] + [format_cell(results[metric].get(col)) for col in columns_order]
        metric_info += ("\t".join(row) + "\n")
    print(metric_info)
    return metric_info


"""
# cd to project root directory
python eval/screenSpot_v2.py --save_path <path_to_save_results>
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="qwen2vl", choices=["qwen2vl", "qwen25vl"])
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/GUI-Actor-2B-Qwen2-VL")
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument('--topk', type=int, default=3, help='Topk')
    parser.add_argument('--no-placeholder', dest='use_placeholder', action='store_false', help='Disable the placeholder')
    parser.set_defaults(use_placeholder=True)

    args = parser.parse_args()

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    pred_path = f"{save_path}/screenspot_v2_all_preds.json"
    metric_path = f"{save_path}/screenspot_v2_all_metrics.txt"

    if os.path.exists(metric_path):
        exit()

    if os.path.exists(pred_path):
        print(f"Loading predictions from {pred_path}")
        with open(pred_path, "r") as f:
            results = json.load(f)
    else:
        print(f"Evaluating {args.model_name_or_path}...")
        results = evaluate(args.model_name_or_path, args.model_type, args.use_placeholder, args.topk)
        with open(pred_path, "w") as f:
            json.dump(results, f)
        print(f"Saved {len(results)} predictions to {pred_path}")

    if not os.path.exists(metric_path):
        metric_info = get_metric(results)
        with open(metric_path, "w") as f:
            f.write(metric_info)
        print(f"Saved metric to {metric_path}")
