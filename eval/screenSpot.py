import torch
from transformers import Qwen2VLProcessor
from datasets import load_dataset
from aguvis.modeling import Qwen2VLForConditionalGenerationWithPointer
from aguvis.decoding_utils import ForceFollowTokensLogitsProcessor
from aguvis.inference import inference, prepare_inputs
from aguvis.utils import draw_bbox, draw_point, do_boxes_overlap
from aguvis.constants import grounding_system_message, DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_END_TOKEN

from PIL import Image
import os
import json
import math
from tqdm import tqdm
import argparse

IMAGE_PATCH_SIZE = 14
POINTER_ONLY = True # if True, will produce grounding feature based on pseudo assistant text output.
MAX_PIXELS = 6585600 # 5860400 = 3220 * 1820. 6585600 = 3360 * 1960

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

def evaluate(model_name_or_path):
    # initialize model
    data_processor = Qwen2VLProcessor.from_pretrained(model_name_or_path)
    tokenizer = data_processor.tokenizer
    for k, v in tokenizer.added_tokens_encoder.items():
        print(v, k)

    model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2"
    ).eval()
    print(f"Loaded model from {model_name_or_path}")

    logits_processor_pointer = ForceFollowTokensLogitsProcessor(
        token_a_id=tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0],
        forced_sequence=[
            tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
        ]
    )

    dataset = load_dataset("rootsautomation/ScreenSpot")["test"]
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
        pred = {
            "file_name": example["file_name"],
            "data_type": example["data_type"],
            "domain": domain_dict[example["data_source"]],
            "instruction": example["instruction"],
            "img_size": example["image"].size,
            "bbox_x1y1x2y2": normalize_bbox(example["bbox"], example["image"].size[0], example["image"].size[1]),
            "pred_points": [],
            "pred_texts": [],
            "is_correct": 0,
            "hit_topk": 0,
        }

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": grounding_system_message
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": example["image"],
                    },
                    {
                        "type": "text",
                        "text": example["instruction"]
                    },
                ],
            },
        ]

        assistant_message = "" if not POINTER_ONLY else "<|im_start|>assistant<|recipient|>os\npyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
        inputs = prepare_inputs(conversation, data_processor, assistant_message)
        _, n_height, n_width = inputs["image_grid_thw"][0]
        output_text, topk_points, topk_values, topk_points_all, attn_scores = inference(model, tokenizer, inputs, data_processor,
                                            logits_processors=[logits_processor_pointer],
                                            pointer_only=POINTER_ONLY,
                                            topk=3,
                                            verbose=False)

        pred["output_text"] = output_text
        pred["topk_points"] = topk_points
        pred["topk_values"] = topk_values
        pred["topk_points_all"] = topk_points_all
        pred["attn_scores"] = attn_scores
        pred["n_width"] = n_width.item()
        pred["n_height"] = n_height.item()
            
        px, py = topk_points[0]

        x1, y1, x2, y2 = pred["bbox_x1y1x2y2"]
        if (x1 <= px <= x2) and (y1 <= py <= y2):
            pred["is_correct"] = 1
            pred["hit_topk"] = 1

        for px, py in topk_points[1:]:
            if (x1 <= px <= x2) and (y1 <= py <= y2):
                pred["hit_topk"] = 1
                break

        results.append(pred)
    
    return results


def get_metric(list_of_examples, 
               domains=["mobile", "desktop", "web"],
               data_types=["text", "icon"]):
    """
    Computes metrics over a list of examples and prints/plots a table.
    
    Each element in list_of_examples is a dict containing:
        - "domain": Domain name (e.g., "web", "mobile", "desktop")
        - "data_type": Data type (e.g., "text", "icon")
        - "is_correct", "is_overlap", "hit_topk", "overlap_topk": binary (0 or 1)
    
    The final table has columns for each domain broken down by UI type (plus a domain-average)
    and overall columns ("All-text", "All-icon", "All-average").
    
    The rows of the table are:
        - is_correct
        - is_overlap
        - hit_topk
        - overlap_topk
    """
    
    # List of metric keys to compute.
    metrics = ["is_correct", "is_overlap", "hit_topk", "overlap_topk"]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/qianhuiwu/blob/qianhuiwu_checkpoints/aguvis_baseline/baseline_same_qwen2vl7binstruct_stage1_ep1_lr1e-05_bs1_mml8192_baseline_same_stage1_all_data_n8_new_6sets/checkpoint-12030")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    
    save_path = f"{model_name_or_path}/final_eval" if args.save_path is None else args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    pred_path = f"{save_path}/screenspot_all_preds.json"
    metric_path = f"{save_path}/screenspot_all_metrics.txt"

    if os.path.exists(metric_path):
        exit()

    if os.path.exists(pred_path):
        print(f"Loading predictions from {pred_path}")
        with open(pred_path, "r") as f:
            results = json.load(f)
    else:
        print(f"Evaluating {model_name_or_path}...")
        results = evaluate(model_name_or_path)
        with open(pred_path, "w") as f:
            json.dump(results, f)
        print(f"Saved {len(results)} predictions to {pred_path}")

    if not os.path.exists(metric_path):
        metric_info = get_metric(results)
        with open(metric_path, "w") as f:
            f.write(metric_info)
        print(f"Saved metric to {metric_path}")
