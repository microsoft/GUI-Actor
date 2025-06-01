import torch
from qwen_vl_utils import process_vision_info
from transformers import LogitsProcessor, LogitsProcessorList
import json
import re
import os

from aguvis.constants import chat_template

def prepare_inputs(messages, data_processor, assistant_message=""):
    """
        example = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": grounding_system_message
                }
            ]
        },
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                    # "image_url": "https://xxxxx.png" or "https://xxxxx.jpg" or "file://xxxxx.png" or "data:image/png;base64,xxxxxxxx", will be split by "base64,"
                },
                {
                    "type": "text",
                    "text": instruction
                },
            ],
        },
        ]
    """
    text = data_processor.apply_chat_template(messages,
                                              tokenize=False,
                                              add_generation_prompt=False,
                                              chat_template=chat_template)
    if assistant_message is not None:
        # e.g., assistant_message = "<|im_start|>assistant<|recipient|>os\npyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
        text += assistant_message

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = data_processor(text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt")

    return inputs


def get_prediction_region_point(attn_scores, n_width, n_height, top_n=30, return_all_regions=True, rect_center=False):
    """
    1. 选出激活的patch（实现了不同的选择方法）
    2. 将连通的patches划分为不同区域
    3. 计算每个区域的平均激活值
    4. 选择平均激活值最高的区域
    5. 返回该区域的中心点作为最终预测点
    """

    # # 方式3: 从中取累计概率达到某个阈值的patch，类似于核采样
    # # 对所有patch按激活值从高到低排序
    # prob_threshold = 0.50
    # sorted_values, sorted_indices = torch.sort(attn_scores[0], descending=True)
    # # 计算归一化的累积概率
    # total_value = torch.sum(attn_scores[0])
    # cumsum = torch.cumsum(sorted_values, dim=0)
    # cumprob = cumsum / total_value
    # # 找到累计概率达到阈值的位置
    # # 注意：最小取1个，最大取top_n个
    # n_indices = 1  # 至少取一个
    # for i, prob in enumerate(cumprob):
    #     if prob >= prob_threshold:
    #         n_indices = i + 1
    #         break
    # # n_indices = min(n_indices, top_n)  # 不超过top_n
    # # 取出满足条件的indices和values
    # topk_indices = sorted_indices[:n_indices]
    # topk_values = sorted_values[:n_indices]

    # 方式2: 激活值大于最大激活值的一定比例的patch
    # 获取最高激活值和阈值
    max_score = attn_scores[0].max().item()
    threshold = max_score * 0.3
    # 选择所有超过阈值的patch
    mask = attn_scores[0] > threshold
    valid_indices = torch.nonzero(mask).squeeze(-1)
    # 如果超过阈值的patch数量多于top_n，则只保留前top_n个
    # if len(valid_indices) > top_n:
    #     # 获取这些激活值并排序
    #     valid_scores = attn_scores[0][valid_indices]
    #     _, sorted_idx = torch.sort(valid_scores, descending=True)
    #     valid_indices = valid_indices[sorted_idx[:top_n]]
    #     topk_values = valid_scores[sorted_idx[:top_n]]
    #     topk_indices = valid_indices
    # else:
    # 否则使用所有超过阈值的patch
    topk_values = attn_scores[0][valid_indices]
    topk_indices = valid_indices

    # # 方式1: 固定取top N个indices及其scores
    # topk_values, topk_indices = attn_scores[0].topk(top_n, dim=-1)
    
    # 将indices转换为2D坐标
    topk_coords = []
    for idx in topk_indices.tolist():
        y = idx // n_width
        x = idx % n_width
        topk_coords.append((y, x, idx))
    
    # 划分连通区域
    regions = []
    visited = set()
    
    for i, (y, x, idx) in enumerate(topk_coords):
        if idx in visited:
            continue
            
        # 开始一个新区域
        region = [(y, x, idx, topk_values[i].item())]
        visited.add(idx)
        queue = [(y, x, idx, topk_values[i].item())]
        
        # BFS查找连通点
        while queue:
            cy, cx, c_idx, c_val = queue.pop(0)
            
            # 检查相邻的4个方向
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                n_idx = ny * n_width + nx
                
                # 检查这个相邻点是否在topk列表中
                for j, (ty, tx, t_idx) in enumerate(topk_coords):
                    if ty == ny and tx == nx and t_idx not in visited:
                        visited.add(t_idx)
                        region.append((ny, nx, t_idx, topk_values[j].item()))
                        queue.append((ny, nx, t_idx, topk_values[j].item()))
        
        regions.append(region)
    
    # 计算每个区域的平均激活值
    region_scores = []
    region_centers = []
    region_points = []
    
    for region in regions:
        # 计算区域平均分数
        avg_score = sum(item[3] for item in region) / len(region)
        region_scores.append(avg_score)

        # 计算每个patch的归一化中心坐标，然后再取平均
        normalized_centers = []
        weights = []
        y_coords = set()
        x_coords = set()

        for y, x, _, score in region:
            # 每个patch的中心点归一化坐标
            center_y = (y + 0.5) / n_height
            center_x = (x + 0.5) / n_width
            normalized_centers.append((center_x, center_y))
            weights.append(score)

            y_coords.add(center_y)
            x_coords.add(center_x)

        region_points.append(normalized_centers)

        # 计算归一化坐标的平均值作为区域中心
        if not rect_center:
            # 加权平均
            total_weight = sum(weights)
            weighted_x = sum(nc[0] * w for nc, w in zip(normalized_centers, weights)) / total_weight
            weighted_y = sum(nc[1] * w for nc, w in zip(normalized_centers, weights)) / total_weight
            avg_center_x, avg_center_y = weighted_x, weighted_y
            # # 直接平均
            # avg_center_x = sum(nc[0] for nc in normalized_centers) / len(normalized_centers)
            # avg_center_y = sum(nc[1] for nc in normalized_centers) / len(normalized_centers)
        else:
            avg_center_x = sum(x_coords) / len(x_coords)
            avg_center_y = sum(y_coords) / len(y_coords)
        region_centers.append((avg_center_x, avg_center_y))
        
    # 选择平均激活值最高的区域
    sorted_indices = sorted(range(len(region_scores)), key=lambda i: region_scores[i], reverse=True)
    sorted_scores = [region_scores[i] for i in sorted_indices]
    sorted_centers = [region_centers[i] for i in sorted_indices]
    sorted_points = [region_points[i] for i in sorted_indices]
    best_point = sorted_centers[0]

    if return_all_regions:
        return best_point, sorted_centers, sorted_scores, sorted_points
    else:
        return best_point


def inference(model, tokenizer, inputs, data_processor, logits_processors: list[LogitsProcessor] = None, pointer_only=False, topk=5, verbose=False):
    inputs = inputs.to(model.device)

    results = model.generate(**inputs,
                            max_new_tokens=2048 if not pointer_only else 1,
                            logits_processor=LogitsProcessorList(logits_processors) if logits_processors else None,
                            return_dict_in_generate=True,
                            output_hidden_states=True,)

    # decode the generated ids
    input_ids = inputs["input_ids"][0]
    generated_ids = results.sequences[0][len(input_ids):]
    input_text = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    n_image_pad_tokens = input_text.count(data_processor.image_token)
    input_text = input_text.replace(data_processor.image_token * n_image_pad_tokens, f"{data_processor.image_token}x{n_image_pad_tokens}")
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    
    if verbose:
        print(f"==> input_text: {input_text}")
        print(f"==> inputs.image_grid_thw: {inputs.image_grid_thw}")
        print(f"==> inputs.pixel_values.shape: {inputs.pixel_values.shape}")
        print(f"==> generated_text: {output_text}")

    # get the hidden states of the "input"/"generated" (pointer_only=True/False) pointer_pad_token_id as decoder vectors
    if not pointer_only:
        decoder_hidden_states = [step_hidden_states[-1][0] for step_hidden_states in results.hidden_states[1:]]
        decoder_hidden_states = torch.cat(decoder_hidden_states, dim=0) # seq_len_generated_ids-1, hidden_size
        pointer_pad_mask = (generated_ids[:-1] == model.config.pointer_pad_token_id) # seq_len_generated_ids-1
    else:
        decoder_hidden_states = results.hidden_states[0][-1][0] # n_all_input_tokens, hidden_size
        pointer_pad_mask = (inputs["input_ids"][0] == model.config.pointer_pad_token_id) # n_all_input_tokens
    decoder_hidden_states = decoder_hidden_states[pointer_pad_mask] # n_pointer_pad_tokens, hidden_size

    if len(decoder_hidden_states) == 0:
        if verbose:
            print("No pointer pad token found in the generated ids")
        return output_text, None, None

    # get the image embeddings as encoder vectors
    image_embeds = model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]) # n_image_tokens, hidden_size

    # attn_scores, _ = model.pointer_head(image_embeds, decoder_hidden_states)
    attn_scores, _ = model.multi_patch_pointer_head(image_embeds, decoder_hidden_states)

    topk_points, topk_values, topk_points_all = None, None, None

    # # 方式1: 返回概率最大的point
    # topk_values, topk_indices = attn_scores[0].topk(topk, dim=-1)
    # # convert image_embed indices to coordinates
    # topk_points = []
    # _, n_height, n_width = (inputs["image_grid_thw"][0] // model.visual.spatial_merge_size).tolist()
    # for idx in topk_indices.tolist():
    #     # point_x = (idx % n_width) / n_width
    #     # point_y = (idx // n_width) / n_height
    #     point_x = (idx % n_width + 0.5) / n_width
    #     point_y = (idx // n_width + 0.5) / n_height
    #     topk_points.append((point_x, point_y))
    # topk_values = topk_values.tolist()

    # 方式2: 按照region划分返回概率最大的point
    _, n_height, n_width = (inputs["image_grid_thw"][0] // model.visual.spatial_merge_size).tolist()
    best_point, region_points, region_scores, region_points_all = get_prediction_region_point(attn_scores, n_width, n_height, return_all_regions=True, rect_center=False)
    topk_points = region_points[:topk] if len(region_points) > topk else region_points
    topk_values = region_scores[:topk] if len(region_scores) > topk else region_scores
    topk_points_all = region_points_all[:topk] if len(region_points_all) > topk else region_points_all

    if verbose:
        for v, p in zip(topk_values, topk_points):
            print(f"{v:.4f} {p}")

    return output_text, topk_points, topk_values, topk_points_all, attn_scores.tolist()


"""python src/aguvis/inference.py"""
if __name__ == "__main__":
    from transformers import Qwen2VLProcessor
    from aguvis.modeling import Qwen2VLForConditionalGenerationWithPointer
    from aguvis.decoding_utils import ForceFollowTokensLogitsProcessor
    from aguvis.constants import DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_END_TOKEN, grounding_system_message
    from aguvis.utils import draw_bbox, draw_point
    from PIL import Image
    
    model_name_or_path = "/home/qianhuiwu/blob/qianhuiwu_checkpoints/aguvis/qwen2vl7binstruct_stage1_ep5_lr0.0001_bs4_mml24576_ufallFalse_ufpnTrue_uflmFalse_ufbmFalse_ufntTrue_ufvFalse_ploss1.0_lmloss-1.0_input_embeds_mlp_all_data"
    POINTER_ONLY = True # whether to use pointer only

    # load model and processor
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

    logits_processor_pointer = ForceFollowTokensLogitsProcessor(
        token_a_id=tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0],
        forced_sequence=[
            tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
        ]
    )

    example = [
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
                    "image": "/home/qianhuiwu/blob/qianhuiwu_main/aguvis/stage1/omniact/images/train_2648.png"
                },
                {
                    "type": "text",
                    "text": "Invite a friend 'abc@example.com' to the event"
                },
            ],
        },
    ]

    assistant_message = "" if not POINTER_ONLY else "<|im_start|>assistant<|recipient|>os\npyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
    inputs = prepare_inputs(example, data_processor, assistant_message)

    output_text, topk_points, topk_values = inference(model, tokenizer, inputs, data_processor,
                                    logits_processors=[logits_processor_pointer],
                                    pointer_only=POINTER_ONLY,
                                    topk=3,
                                    verbose=True)

    if topk_points is not None:
        image = Image.open(example[1]["content"][0]["image"])
        image_width, image_height = image.size
        point_x, point_y = topk_points[0]

        display_image = draw_point(image, (point_x * image_width, point_y * image_height), color='green')

        for i, (point_x, point_y) in enumerate(topk_points[1:]):
            display_image = draw_point(display_image, (point_x * image_width, point_y * image_height), color='gray')
        
        display_image.save("tmp_results.png")
