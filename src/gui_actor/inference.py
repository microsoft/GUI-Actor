import torch
import json
import re
import os
from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2VLForConditionalGeneration,
    LogitsProcessor,
    LogitsProcessorList,
    AutoModelForCausalLM,
    AutoTokenizer
)
from gui_actor.constants import (
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    chat_template
)

class ForceFollowTokensLogitsProcessor(LogitsProcessor):
    """
    Forces tokens B (pointer_pad_token) and C (pointer_end_token) to follow token A (pointer_start_token).
    Whenever token_a_id is generated, enqueue the forced_sequence (e.g. [B, C]).
    As long as forced tokens remain in the queue, force them in the output.
    """
    def __init__(self, token_a_id, forced_sequence=[DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_END_TOKEN]):
        super().__init__()
        self.token_a_id = token_a_id
        self.forced_sequence = forced_sequence  # list of token IDs, e.g. [B_id, C_id]
        self.force_queue = []  # holds the tokens we still need to force

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Called at each decoding step to modify `scores`.
        
        Args:
            input_ids: shape (batch_size, seq_len). The already-decoded tokens.
            scores:    shape (batch_size, vocab_size). Model logits for the next token.
        """
        batch_size = input_ids.shape[0]
        if batch_size > 1:
            raise NotImplementedError("Batch size must be 1 for this logits processor.")
        
        # We assume batch_size=1 for simplicity; if you have multiple sequences,
        # you'll need to adapt the logic to handle each item in the batch.
        last_token_id = input_ids[0, -1].item()

        # If the last token was A, enqueue B and C
        if last_token_id == self.token_a_id:
            self.force_queue.extend(self.forced_sequence)
        
        # If we have forced tokens waiting in the queue, override the distribution
        if len(self.force_queue) > 0:
            forced_token = self.force_queue.pop(0)  # next token to force
            # Create a mask of -inf for all tokens except the forced one
            new_scores = torch.full_like(scores, float('-inf'))
            new_scores[0, forced_token] = 0.0  # log prob = 0 => prob = 1
            return new_scores
        
        # Otherwise, return scores unmodified
        return scores


def get_prediction_region_point(attn_scores, n_width, n_height, top_n=30, activation_threshold=0.3, return_all_regions=True, rect_center=False):
    """
    1. Select activated patches
    2. Divide connected patches into different regions
    3. Calculate the average activation value for each region
    4. Select the region with the highest average activation value
    5. Return the center point of that region as the final prediction point
    """

    # Get patches with activation values greater than a certain proportion of the maximum activation value as activated patches
    # Get the highest activation value and threshold
    max_score = attn_scores[0].max().item()
    threshold = max_score * activation_threshold
    # Select all patches above the threshold
    mask = attn_scores[0] > threshold
    valid_indices = torch.nonzero(mask).squeeze(-1)
    topk_values = attn_scores[0][valid_indices]
    topk_indices = valid_indices
    
    # Convert indices to 2D coordinates
    topk_coords = []
    for idx in topk_indices.tolist():
        y = idx // n_width
        x = idx % n_width
        topk_coords.append((y, x, idx))
    
    # Divide into connected regions
    regions = []
    visited = set()
    for i, (y, x, idx) in enumerate(topk_coords):
        if idx in visited:
            continue
            
        # Start a new region
        region = [(y, x, idx, topk_values[i].item())]
        visited.add(idx)
        queue = [(y, x, idx, topk_values[i].item())]
        
        # BFS to find connected points
        while queue:
            cy, cx, c_idx, c_val = queue.pop(0)
            
            # Check 4 adjacent directions
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                n_idx = ny * n_width + nx
                
                # Check if this adjacent point is in the topk list
                for j, (ty, tx, t_idx) in enumerate(topk_coords):
                    if ty == ny and tx == nx and t_idx not in visited:
                        visited.add(t_idx)
                        region.append((ny, nx, t_idx, topk_values[j].item()))
                        queue.append((ny, nx, t_idx, topk_values[j].item()))
        
        regions.append(region)
    
    # Calculate the average activation value for each region
    region_scores = []
    region_centers = []
    region_points = []
    
    for region in regions:
        # Calculate average score for the region
        avg_score = sum(item[3] for item in region) / len(region)
        region_scores.append(avg_score)

        # Calculate normalized center coordinates for each patch, then take the average
        normalized_centers = []
        weights = []
        y_coords = set()
        x_coords = set()

        for y, x, _, score in region:
            # Normalized coordinates of the center point for each patch
            center_y = (y + 0.5) / n_height
            center_x = (x + 0.5) / n_width
            normalized_centers.append((center_x, center_y))
            weights.append(score)

            y_coords.add(center_y)
            x_coords.add(center_x)

        region_points.append(normalized_centers)

        # Calculate the average of normalized coordinates as the region center
        if not rect_center:
            # Weighted average
            total_weight = sum(weights)
            weighted_x = sum(nc[0] * w for nc, w in zip(normalized_centers, weights)) / total_weight
            weighted_y = sum(nc[1] * w for nc, w in zip(normalized_centers, weights)) / total_weight
            avg_center_x, avg_center_y = weighted_x, weighted_y
            # # Simple average
            # avg_center_x = sum(nc[0] for nc in normalized_centers) / len(normalized_centers)
            # avg_center_y = sum(nc[1] for nc in normalized_centers) / len(normalized_centers)
        else:
            avg_center_x = sum(x_coords) / len(x_coords)
            avg_center_y = sum(y_coords) / len(y_coords)
        region_centers.append((avg_center_x, avg_center_y))
        
    # Select the region with the highest average activation value
    sorted_indices = sorted(range(len(region_scores)), key=lambda i: region_scores[i], reverse=True)
    sorted_scores = [region_scores[i] for i in sorted_indices]
    sorted_centers = [region_centers[i] for i in sorted_indices]
    sorted_points = [region_points[i] for i in sorted_indices]
    best_point = sorted_centers[0]

    if return_all_regions:
        # Outputs:
        # 1. best_point: the center point of the region with the highest average activation value
        # 2. sorted_centers: the center points of all regions, sorted by the average activation value in descending order
        # 3. sorted_scores: the average activation values of all regions, sorted in descending order
        # 4. sorted_points: the normalized center coordinates of all patches, sorted by the average activation value in descending order
        return best_point, sorted_centers, sorted_scores, sorted_points
    else:
        return best_point


def inference(conversation, model, tokenizer, data_processor, logits_processor=None, use_placeholder=False, topk=5):
    """
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
    """
    if logits_processor is None:
        logits_processor = ForceFollowTokensLogitsProcessor(
            token_a_id=tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0],
            forced_sequence=[
                tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
            ]
        )
    
    assiatant_starter = "" if not use_placeholder else "<|im_start|>assistant<|recipient|>os\npyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"

    pred = {
        "output_text": None, # generated text
        "n_width": None, # number of patch_tokens in width dimension
        "n_height": None, # number of patch_tokens in height dimension
        "attn_scores": None, # attention scores over the image patches
        "topk_points": None, # topk points
        "topk_values": None, # topk values
        "topk_points_all": None, # all points
    }

    # prepare text
    text = data_processor.apply_chat_template(conversation,
                                            tokenize=False,
                                            add_generation_prompt=False,
                                            chat_template=chat_template
                                            )
    text += assiatant_starter

    # prepare inputs
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = data_processor(text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt"
                            )
    inputs = inputs.to(model.device)

    # generate
    results = model.generate(**inputs,
                            max_new_tokens=2048 if not use_placeholder else 1,
                            logits_processor=LogitsProcessorList([logits_processor]),
                            return_dict_in_generate=True,
                            output_hidden_states=True
                            )


    # decode the generated ids
    input_ids = inputs["input_ids"][0]
    generated_ids = results.sequences[0][len(input_ids):]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    pred["output_text"] = output_text

    # check if there are <POINTER_TOKEN> is inside the input_ids or generated_ids
    if use_placeholder:
        pointer_pad_mask = (inputs["input_ids"][0] == model.config.pointer_pad_token_id) # n_all_input_tokens
    else:
        pointer_pad_mask = (generated_ids[:-1] == model.config.pointer_pad_token_id) # seq_len_generated_ids-1

    # if there are no <POINTER_TOKEN> in the input_ids or generated_ids, return the pred
    if len(pointer_pad_mask) == 0:
        return pred
    
    # otherwise, get the coordinate from the action head
    if use_placeholder:
        decoder_hidden_states = results.hidden_states[0][-1][0] # n_all_input_tokens, hidden_size
    else:
        decoder_hidden_states = [step_hidden_states[-1][0] for step_hidden_states in results.hidden_states[1:]]
        decoder_hidden_states = torch.cat(decoder_hidden_states, dim=0) # seq_len_generated_ids-1, hidden_size
    decoder_hidden_states = decoder_hidden_states[pointer_pad_mask] # n_pointer_pad_tokens, hidden_size

    # get the image embeddings as encoder vectors
    # image_embeds = model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]) # n_image_tokens, hidden_size
    image_mask = (inputs["input_ids"][0] == tokenizer.encode("<|image_pad|>")[0])
    image_embeds = results.hidden_states[0][0][0][image_mask] # n_image_tokens, hidden_size

    attn_scores, _ = model.multi_patch_pointer_head(image_embeds, decoder_hidden_states)
    pred["attn_scores"] = attn_scores.tolist()

    _, n_height, n_width = (inputs["image_grid_thw"][0] // model.visual.spatial_merge_size).tolist()
    pred["n_width"] = n_width
    pred["n_height"] = n_height

    # get the topk points according to the attention scores
    best_point, region_points, region_scores, region_points_all = get_prediction_region_point(attn_scores, n_width, n_height, return_all_regions=True, rect_center=False)
    topk_points = region_points[:topk] if len(region_points) > topk else region_points
    topk_values = region_scores[:topk] if len(region_scores) > topk else region_scores
    topk_points_all = region_points_all[:topk] if len(region_points_all) > topk else region_points_all
    pred["topk_points"] = topk_points
    pred["topk_values"] = topk_values
    pred["topk_points_all"] = topk_points_all

    return pred