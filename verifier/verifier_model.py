import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import json
import re
import os
import tempfile
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from typing import List, Literal, Optional
import numpy as np
import random

grounding_system_message = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name


def draw_point_list(img, points, color='red', size=1, crop=True, sample_crop=False, crop_size=500):
    draw = ImageDraw.Draw(img)
    radius = np.ceil(7 * size).astype(int)
    for point in points:
        circle_bbox = [
            point[0] - radius,  # x1
            point[1] - radius,  # y1
            point[0] + radius,  # x2
            point[1] + radius   # y2
        ]
        draw.ellipse(circle_bbox, outline=color, width=np.ceil(3 * size).astype(int))


    if crop:
        x, y = points[0]
        width, height = img.size 
        crop_half_size = crop_size         
        left = max(0, x - crop_half_size)
        right = min(width-1, x + crop_half_size)
        top = max(0, y - crop_half_size)
        bottom = min(height-1, y + crop_half_size)
        try:
            img = img.crop((left, top, right, bottom))
        except Exception as e:
            print(f"Error cropping image: {e}")
            # If cropping fails, return the original image
            return img
    return img



class GroundingVerifier():
    def __init__(self,
        model_name_or_path="microsoft/GUI-Actor-Verifier-2B",
        json_prediction=None,
        method='score' # 'best_one', 'comparison', 'score'
    ):
        self.method = method
        self.model_name_or_path = model_name_or_path
        self.system_message = {
                                "role": "system",
                                "content": grounding_system_message,
                            }
        self.json_prediction_path = json_prediction
        # load json prediction
        assert os.path.exists(json_prediction) and os.path.isfile(json_prediction), "Invalid json prediction path."
        with open(json_prediction, 'r') as f:
            self.json_prediction = json.load(f)
        
        self.verifier_crop_size = 500 # half of the true crop size
        # use 0.95 for ss-pro 
        if '-pro' in self.json_prediction_path.lower(): 
            self.threshold = 0.95
        else: # use 0.8 for ss and ss-v2
            self.threshold = 0.8 
        
        self.json_index_dict = {}
        for i, item in enumerate(self.json_prediction):
            key = 'img_filename' if 'img_filename' in item else 'file_name'
            json_key = item[key] + item['instruction'] if 'instruction' in item else ''
            self.json_index_dict[json_key] = i



    def load_model(self, verifier_path):
        if self.method == 'best_one':
            return
        else:
            verifier_model_name_or_path = verifier_path

        self.verifier = Qwen2VLForConditionalGeneration.from_pretrained(
            verifier_model_name_or_path,
            device_map="cuda:0",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        self.verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_model_name_or_path, trust_remote_code=True)
        self.verifier_processor = AutoProcessor.from_pretrained(verifier_model_name_or_path)
        self.verifier_processor.tokenizer.pad_token = self.verifier_processor.tokenizer.eos_token   



    def set_generation_config(self, **kwargs):
        pass


    def verify(self, instruction, image):
        verifier_prompt = "Please observe the screenshot and exame whether the hollow red circle accurately placed on the intended position in the image: '{}'. Answer True or False."
        full_prompt = verifier_prompt.format(instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]
        text_input = self.verifier_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.verifier_processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:0")


        # get the token probability of True and False using the verifier
        # Forward pass to get logits
        with torch.no_grad():
            outputs = self.verifier(**inputs)
            logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)

        # Get the last token's logits
        last_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)


        # Get vocab IDs for "True" and "False"
        true_id = self.verifier_processor.tokenizer.encode("True", add_special_tokens=False)[0]
        false_id = self.verifier_processor.tokenizer.encode("False", add_special_tokens=False)[0]

        # Get probabilities using softmax
        probs = torch.softmax(last_token_logits, dim=-1)
        true_prob = probs[0, true_id].item()
        false_prob = probs[0, false_id].item()
        score = true_prob / (true_prob + false_prob)
        return score



    def verifier_score(self, instruction, image, box):
        box = [box]
        img_copy = image.copy()
        img_new = draw_point_list(img_copy, box, crop_size=self.verifier_crop_size) 
        score = self.verify(instruction, img_new)
        return score


    def get_prediction_region_point(self, attn_scores, n_width, n_height, top_n=20, return_all_regions=True, rect_center=False, no_groups=False):
        attn_scores = np.array(attn_scores)
        max_score = attn_scores.max()
        threshold = max_score * 0.2 
        # select patches with activation scores above the threshold
        mask = attn_scores > threshold
        valid_indices = np.where(mask)
        # keep only top_n patches
        if len(valid_indices[1]) > top_n:
            valid_scores = attn_scores[valid_indices]
            sorted_idx = np.argsort(valid_scores)[::-1][:top_n]
            valid_indices = valid_indices[1][sorted_idx]
            topk_values = valid_scores[sorted_idx]
            topk_indices = valid_indices
        else:
            topk_values = attn_scores[valid_indices].tolist()
            topk_indices = valid_indices[1]

        # topk_values, topk_indices = attn_scores.topk(top_n, dim=-1)
        if n_width * n_height != attn_scores.shape[1]:
            n_width = n_width // 2
            n_height = n_height // 2


        # transform the topk_indices into coordinates
        topk_coords = []
        for idx in topk_indices:
            x = idx % n_width
            y = idx // n_width
            topk_coords.append((int(y), int(x), int(idx)))
        
        # divide the topk_coords into regions based on connectivity
        regions = []
        visited = set()
        
        for i, (y, x, idx) in enumerate(topk_coords):
            if idx in visited:
                continue
                
            region = [(y, x, idx, topk_values[i])]
            visited.add(idx)
            queue = [(y, x, idx, topk_values[i])]
            
            # BFS
            while queue:
                cy, cx, c_idx, c_val = queue.pop(0)
                
                # check four directions
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx
                    n_idx = ny * n_width + nx
                    
                    # check whether the new coordinates are within bounds
                    for j, (ty, tx, t_idx) in enumerate(topk_coords):
                        if ty == ny and tx == nx and t_idx not in visited:
                            visited.add(t_idx)
                            region.append((ny, nx, t_idx, topk_values[j]))
                            queue.append((ny, nx, t_idx, topk_values[j]))
            
            regions.append(region)
        
        region_scores = []
        region_centers = []
        region_points = []
        
        for region in regions:
            # calculate the average score of the region
            avg_score = sum(item[3] for item in region) / len(region)
            region_scores.append(avg_score)


            # calculate the normalized center of the region
            normalized_centers = []
            weights = []
            y_coords = set()
            x_coords = set()

            for y, x, _, score in region:
                center_y = (y + 0.5) / n_height
                center_x = (x + 0.5) / n_width
                normalized_centers.append((center_x, center_y))
                weights.append(score)


                y_coords.add(center_y)
                x_coords.add(center_x)


            region_points.append(normalized_centers)


            # calculate the average center of the region
            if not rect_center:
                # weighted average
                total_weight = sum(weights)
                weighted_x = sum(nc[0] * w for nc, w in zip(normalized_centers, weights)) / total_weight
                weighted_y = sum(nc[1] * w for nc, w in zip(normalized_centers, weights)) / total_weight
                avg_center_x, avg_center_y = weighted_x, weighted_y
            else:
                avg_center_x = sum(x_coords) / len(x_coords)
                avg_center_y = sum(y_coords) / len(y_coords)
            region_centers.append((avg_center_x, avg_center_y))
            
        # select top regions based on scores
        sorted_indices = sorted(range(len(region_scores)), key=lambda i: region_scores[i], reverse=True)
        sorted_scores = [region_scores[i] for i in sorted_indices]
        sorted_centers = [region_centers[i] for i in sorted_indices]
        sorted_points = [region_points[i] for i in sorted_indices]
        best_point = sorted_centers[0]


        if no_groups:
            if return_all_regions:
                return sorted_centers + [[(x[1] + 0.5) / n_width, (x[0] + 0.5) /n_height] for x in topk_coords] 
            else:
                return sorted_centers + [(topk_coords[0][1]+ 0.5) / n_width, (topk_coords[0][0]+ 0.5) / n_height]

        if return_all_regions:
            return best_point, sorted_centers, sorted_scores, sorted_points
        else:
            return best_point




    def ground_only_positive(self, instruction, image, target_point):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        else:
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        
        width, height = image.size
        
        print(image_path)
        if 'v2' in image_path:
            key = image_path.split('/')[-1]
        elif 'Pro' in image_path:
            key = '/'.join(image_path.split('/')[-2:])
        else:
            key = image_path.split('/')[-1]
        key += instruction
        index = self.json_index_dict[key]
 

        if self.method == 'best_one':
            predictions = self.json_prediction[index]['topk_points']
            predictions = [predictions[0]] # only the first one
        else:
            attn_scores = self.json_prediction[index]['attn_scores']
            if 'n_width' in self.json_prediction[index]:
                n_width, n_height = self.json_prediction[index]['n_width'], self.json_prediction[index]['n_height']
            elif 'img_size_crop' in self.json_prediction[index]:
                n_width, n_height = self.json_prediction[index]['img_size_crop']
            else:
                raise ValueError("Invalid json prediction format. 'n_width' or 'img_size_crop' not found.")
            predictions = self.get_prediction_region_point(attn_scores, n_width, n_height, top_n=20, return_all_regions=True, rect_center=False, no_groups=True)

        pred_points_list = [[pred[0]  * image.size[0], pred[1]  * image.size[1]] for pred in predictions]
        score_list = []


        print(predictions, len(predictions))
        if len(predictions) > 1:
            if self.method == 'score':
                for point in pred_points_list[len(score_list):]:
                    score = self.verifier_score(instruction, image, point)
                    score_list.append(score)
                    if score >= self.threshold: 
                        break
                # get the max score
                print(score_list, len(score_list))
                point = predictions[score_list.index(max(score_list))]
        else:
            point = predictions[0]
         

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": pred_points_list,
            "bbox": None,
            "point": point,
        }
        return result_dict







