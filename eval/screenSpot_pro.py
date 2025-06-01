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
from tqdm import tqdm
import argparse

POINTER_ONLY = True # if True, will produce grounding feature based on pseudo assistant text output.

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

def evaluate(model_name_or_path, data_fn, image_dir, resize_to_pixels=None):
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

    # load data
    with open(data_fn, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {data_fn}")

    results = []
    for example in tqdm(data, total=len(data)):
        pred = {
            "file_name": example["img_filename"],
            "ui_type": example["ui_type"],
            "group": example["group"],
            "platform": example["platform"],
            "application": example["application"],
            "id": example["id"],
            "instruction": example["instruction"],
            "img_size": example["img_size"],
            "bbox_x1y1x2y2": normalize_bbox(example["bbox"], example["img_size"][0], example["img_size"][1]),
            "pred_points": [],
            "pred_texts": [],
            "is_correct": 0,
            "hit_topk": 0,
        }
        
        image_width, image_height = example["img_size"]
        bbox_x1y1x2y2 = example["bbox"]

        image_path = os.path.join(image_dir, example["img_filename"])
        image = Image.open(image_path)

        if (resize_to_pixels is not None) and ((image_width * image_height) != resize_to_pixels):
            resize_ratio = (resize_to_pixels / (image_width * image_height)) ** 0.5
            image_width_resized, image_height_resized = int(image_width * resize_ratio), int(image_height * resize_ratio)
            image_width, image_height = image_width_resized, image_height_resized
            image = image.resize((image_width_resized, image_height_resized))
            bbox_x1y1x2y2 = [bbox_x1y1x2y2[0] * resize_ratio, bbox_x1y1x2y2[1] * resize_ratio, bbox_x1y1x2y2[2] * resize_ratio, bbox_x1y1x2y2[3] * resize_ratio]
            example["img_size_resized"] = [image_width_resized, image_height_resized]
        else:
            example["img_size_resized"] = None
        
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
                        "image": image,
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
               groups=["Dev", "Creative", "CAD", "Scientific", "Office", "OS"],
               ui_types=["text", "icon"]):
    """
    Computes metrics over a list of examples and prints/plots a table.
    
    Each element in list_of_examples is a dict containing:
        - "group": Group name (e.g., "Dev", "Creative", etc.)
        - "ui_type": UI type (e.g., "text", "icon")
        - "is_correct", "is_overlap", "hit_topk", "overlap_topk": binary (0 or 1)
    
    The final table has columns for each group broken down by UI type (plus a group-average)
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
    for group in groups:
        # Filter examples for the current group.
        group_examples = [ex for ex in list_of_examples if ex.get("group") == group]
        for ui in ui_types:
            # Filter further for the specific UI type.
            group_ui_examples = [ex for ex in group_examples if ex.get("ui_type") == ui]
            col_name = f"{group}-{ui}"
            for metric in metrics:
                results[metric][col_name] = compute_mean(group_ui_examples, metric)
        
        # Compute group-average (all UI types for this group).
        col_name_avg = f"{group}-avg"
        for metric in metrics:
            results[metric][col_name_avg] = compute_mean(group_examples, metric)

    # Compute overall metrics for each UI type across all groups.
    for ui in ui_types:
        ui_examples = [ex for ex in list_of_examples if ex.get("ui_type") == ui]
        col_name = f"All-{ui}"
        for metric in metrics:
            results[metric][col_name] = compute_mean(ui_examples, metric)
    
    # Compute overall average across all examples.
    overall_key = "All-avg"
    for metric in metrics:
        results[metric][overall_key] = compute_mean(list_of_examples, metric)
    
    # Define the order of columns.
    columns_order = []
    for group in groups:
        for ui in ui_types:
            columns_order.append(f"{group}-{ui}")
        columns_order.append(f"{group}-avg")
    for ui in ui_types:
        columns_order.append(f"All-{ui}")
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
    metric_info += header_tab + "\n"
    # Each row.
    for metric in metrics:
        row = [metric] + [format_cell(results[metric].get(col)) for col in columns_order]
        metric_info += ("\t".join(row) + "\n")
    print(metric_info)
    return metric_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/qianhuiwu/blob/qianhuiwu_checkpoints/aguvis_baseline/baseline_same_qwen2vl7binstruct_stage1_ep1_lr1e-05_bs1_mml8192_baseline_same_stage1_all_data_n8_new_6sets/checkpoint-12030")
    parser.add_argument("--data_path", type=str, default="/mnt/data/ScreenSpot-Pro")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--resize_to_pixels", type=int, default=3200*1800, help="If set to <0, will not resize the image.")
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    resize_to_pixels = args.resize_to_pixels if args.resize_to_pixels > 0 else None

    image_dir = os.path.join(args.data_path, "images")
    data_fn = os.path.join(args.data_path, "annotations/all.json")
    
    save_path = f"{model_name_or_path}/final_eval" if args.save_path is None else args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    pred_path = f"{save_path}/screenspot-Pro_all_preds_StandardResize.json"
    metric_path = f"{save_path}/screenspot-Pro_all_preds_StandardResize.txt"

    if os.path.exists(metric_path):
        exit()

    if os.path.exists(pred_path):
        print(f"Loading predictions from {pred_path}")
        with open(pred_path, "r") as f:
            results = json.load(f)
    else:
        print(f"Evaluating {model_name_or_path}...")
        results = evaluate(model_name_or_path, data_fn, image_dir, resize_to_pixels)
        with open(pred_path, "w") as f:
            json.dump(results, f)
        print(f"Saved {len(results)} predictions to {pred_path}")

    if not os.path.exists(metric_path):
        metric_info = get_metric(results)
        with open(metric_path, "w") as f:
            f.write(metric_info)
        print(f"Saved metric to {metric_path}")