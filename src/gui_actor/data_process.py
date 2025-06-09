# Process datasets
# Format according to pyautogui, and add a bbox key for use in dataset.py
import json
import os
from tqdm import tqdm
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import re
from collections import Counter


def is_bbox_valid(item):
    """
    Check if all bboxes in an item are valid
    Valid bboxes meet the following conditions:
    1. Coordinates are in the range [0,1]
    2. left < right, top < bottom
    
    Args:
        item: Data item containing conversations
        
    Returns:
        bool: True if all bboxes are valid, False otherwise
    """
    for ele in item["conversations"]:
        if ele["from"] == "human":
            continue
        
        if "bbox_gt" not in ele:
            continue
            
        ele_bbox = ele["bbox_gt"]
        # Check if bbox is normal [left, top, right, bottom]
        if (ele_bbox[0] < 0 or ele_bbox[1] < 0 or 
            ele_bbox[2] > 1 or ele_bbox[3] > 1 or
            ele_bbox[0] >= ele_bbox[2] or ele_bbox[1] >= ele_bbox[3]):
            print(f"Abnormal bbox: {ele_bbox}")
            return False
    
    return True


def visualize_element(img_path, bbox, center, instruction):
    try:
        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        # Get image dimensions
        img_width, img_height = pil_img.size
        
        # Convert normalized coordinates to pixel coordinates
        bbox_pixel = [
            int(bbox[0] * img_width),
            int(bbox[1] * img_height),
            int(bbox[2] * img_width),
            int(bbox[3] * img_height)
        ]
        center_pixel = [
            int(center[0] * img_width),
            int(center[1] * img_height)
        ]
        
        font_size = 40
        font = ImageFont.load_default()
        
        # Draw bounding box, center point, and instruction on the image
        draw.rectangle(bbox_pixel, outline=(255, 0, 0), width=2)  # Red bounding box
        draw.ellipse((center_pixel[0]-5, center_pixel[1]-5, center_pixel[0]+5, center_pixel[1]+5), fill=(0, 255, 0))  # Green center point
        
        # Draw text above the bounding box
        text_position = (bbox_pixel[0], max(0, bbox_pixel[1] - 25))
        # Truncate long instructions
        short_instruction = instruction[:60] + "..." if len(instruction) > 60 else instruction
        
        text = short_instruction
   
        draw.text(text_position, text, fill=(255, 0, 0), font=font)
        
        return pil_img
    
    except Exception as e:
        print(f"Error during visualization: {e}")
        return None


# # SeeClick
# json_path = '/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/SeeClick/seeclick_web.json'
# img_dir = '/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/SeeClick/cpfs01/user/chengkanzhi/seeclick_web_imgs'

# # vis_dir = '/root/bayes-tmp/chengkz/datasets/Seeclick-data/visualization'  # Path to save visualization results
# # os.makedirs(vis_dir, exist_ok=True)  # Ensure directory exists

# with open(json_path, 'r') as f:
#     data = json.load(f)

# seeclick_data = []
# ele_num = 0
# random.shuffle(data)
# for item in tqdm(data):
#     # print(item)
#     # input()

#     img_filename = item["img_filename"]
#     img_path = os.path.join(img_dir, img_filename)
#     # if not os.path.exists(img_path):
#     #     print(f"img_path not exists: {img_path}")
#     #     input()

#     conversation = []
#     elements = item["elements"]
#     random.shuffle(elements)
#     elements = elements[:15]
#     item_ele_num = 0  # Used to count the number of valid elements in the current data item
#     for i, ele in enumerate(elements):
#         instruction = ele["instruction"]
#         if i == 0:
#             instruction_input = f"<image> {instruction}"
#         else:
#             instruction_input = instruction

#         bbox = ele["bbox"]  # [left, top, right, bottom]
#         center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
#         x, y = center[0], center[1]
#         action_input = f"pyautogui.click(x={x:.4f}, y={y:.4f})"
        
#         # # Use function to create visualization
#         # vis_img = visualize_element(img_path, bbox, center, instruction)
#         # if vis_img:
#         #     # Save visualization result for single element
#         #     ele_vis_path = os.path.join(vis_dir, f"vis_{img_filename.split('.')[0]}_ele{i+1}.jpg")
#         #     vis_img.save(ele_vis_path)
        
#         conversation.append({
#             "from": "human",
#             "value": instruction_input
#         })
        
#         conversation.append({
#             "from": "gpt", 
#             "value": action_input,
#             "recipient": "os",
#             "end_turn": True,
#             "bbox_gt": bbox
#         })
#         item_ele_num += 1
    
#     data_item = {
#         "image": img_filename,
#         "conversations": conversation
#     }
    
#     # Use function to check if bbox is valid
#     if is_bbox_valid(data_item):
#         seeclick_data.append(data_item)
#         ele_num += item_ele_num  # Only count elements when the data item is valid
#     else:
#         print(f"Discarding invalid SeeClick data item: {img_filename}")
# print(f"Seeclick_data length: {len(seeclick_data)}")
# print(f"Seeclick_data ele_num: {ele_num}")
# json.dump(seeclick_data, open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/SeeClick/seeclick_aguvis_bbox.json", "w"), indent=4)
# print("Success")


# # AMEX
# json_path = '/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/AMEX/amex_raw.json'
# img_dir = '/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/AMEX/screenshots'
# with open(json_path, 'r') as f:
#     data = json.load(f)

# # vis_dir = '/root/bayes-tmp/chengkz/datasets/AMEX/visualization'  # Path to save visualization results
# # os.makedirs(vis_dir, exist_ok=True)  # Ensure directory exists

# amex_data = []
# ele_num = 0
# random.shuffle(data)
# for item in tqdm(data):
    
#     img_filename = item["img_filename"].split('/')[-1]
#     img_path = os.path.join(img_dir, img_filename)
#     # if not os.path.exists(img_path):
#     #     print(f"img_path not exists: {img_path}")
#     #     input()

#     conversation = []
#     elements = item["elements"]
#     random.shuffle(elements)
#     item_ele_num = 0  # Used to count the number of valid elements in the current data item
#     for i, ele in enumerate(elements):

#         instruction = ele["instruction"]
#         if i == 0:
#             instruction_input = f"<image> {instruction}"
#         else:
#             instruction_input = instruction

#         bbox = ele["bbox"]  # [left, top, right, bottom]
#         center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
#         x, y = center[0], center[1]
#         action_input = f"pyautogui.click(x={x:.4f}, y={y:.4f})"

#         # # Use function to create visualization
#         # vis_img = visualize_element(img_path, bbox, center, instruction)
#         # if vis_img:
#         #     # Save visualization result for single element
#         #     ele_vis_path = os.path.join(vis_dir, f"vis_{img_filename.split('.')[0]}_ele{i+1}.jpg")
#         #     vis_img.save(ele_vis_path)
        
#         conversation.append({
#             "from": "human",
#             "value": instruction_input
#         })
        
#         conversation.append({
#             "from": "gpt", 
#             "value": action_input,
#             "recipient": "os",
#             "end_turn": True,
#             "bbox_gt": bbox
#         })
#         item_ele_num += 1
    
#     data_item = {
#         "image": img_filename,
#         "conversations": conversation
#     }
    
#     # Use function to check if bbox is valid
#     if is_bbox_valid(data_item):
#         amex_data.append(data_item)
#         ele_num += item_ele_num  # Only count elements when the data item is valid

# print(f"AMEX_data length: {len(amex_data)}")
# print(f"AMEX_data ele_num: {ele_num}")
# json.dump(amex_data, open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/AMEX/amex_aguvis_bbox.json", "w"), indent=4)
# print("Success")


# # Wave-UI
# from datasets import load_dataset
# import uuid

# # Set save path
# wave_img_dir = '/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/Wave-UI/images_fixed'
# os.makedirs(wave_img_dir, exist_ok=True)  # Ensure directory exists

# # # Visualization path
# # wave_vis_dir = '/root/bayes-tmp/chengkz/datasets/Wave-UI/visualization'
# # os.makedirs(wave_vis_dir, exist_ok=True)  # Ensure directory exists

# # Load dataset
# dataset = load_dataset("agentsea/wave-ui", streaming=True)

# # Prepare to store data
# wave_data = []
# ele_num = 0

# num_omniact = 0
# num_mind2web_test = 0
# num_screenspot = 0
# for example in tqdm(dataset["train"]):

#     if "omniact" in example["source"]:
#         num_omniact += 1
#         continue
#     if "mind2web_test" in example["source"]:
#         num_mind2web_test += 1
#         continue
#     if "screenspot" in example["source"]:
#         num_screenspot += 1
#         continue
    
#     # 1. Save image
#     img = example['image']
#     img_id = str(uuid.uuid4())  # Generate unique ID as filename
#     img_filename = f"{img_id}.png"
#     img_path = os.path.join(wave_img_dir, img_filename)
#     img.save(img_path)
    
#     # 2. Prepare data item
#     instruction = example['name']
#     if example['OCR'] is not None:
#         instruction += ' OCR: ' + example['OCR']
#     resolution = example['resolution']
    
#     # 3. Convert absolute coordinates to normalized coordinates in [0,1] range
#     bbox_abs = example['bbox']  # [left, top, right, bottom] absolute coordinates
#     bbox_norm = [
#         bbox_abs[0] / resolution[0],  # left
#         bbox_abs[1] / resolution[1],  # top
#         bbox_abs[2] / resolution[0],  # right
#         bbox_abs[3] / resolution[1]   # bottom
#     ]
    
#     # Calculate center point (normalized coordinates)
#     center_norm = [
#         (bbox_norm[0] + bbox_norm[2]) / 2,  # x
#         (bbox_norm[1] + bbox_norm[3]) / 2   # y
#     ]

#     # Build conversation format
#     instruction_input = f"<image> {instruction}"
#     x, y = center_norm[0], center_norm[1]
#     action_input = f"pyautogui.click(x={x:.4f}, y={y:.4f})"
    
#     # Create conversation list
#     conversation = [
#         {
#             "from": "human",
#             "value": instruction_input
#         },
#         {
#             "from": "gpt", 
#             "value": action_input,
#             "recipient": "os",
#             "end_turn": True,
#             "bbox_gt": bbox_norm
#         }
#     ]
    
#     # Add to dataset
#     data_item = {
#         "image": img_filename,
#         "conversations": conversation
#     }
    
#     # Use function to check if bbox is valid
#     if is_bbox_valid(data_item):
#         wave_data.append(data_item)
#         ele_num += 1
#     else:
#         print(f"Discarding invalid Wave-UI data item: {img_filename}")
    
#     # # Optional: Create visualization result
#     # vis_img = visualize_element(img_path, bbox_norm, center_norm, instruction)
#     # if vis_img:
#     #     vis_path = os.path.join(wave_vis_dir, f"vis_{img_filename}")
#     #     vis_img.save(vis_path)
    
#     # sample_count += 1

# # Save processed data as JSON file
# random.shuffle(wave_data)
# json.dump(wave_data, open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/Wave-UI/wave_ui_aguvis_bbox_fixed.json", "w"), indent=4)
# print(f"Sample count: {len(wave_data)}")
# print(f"Element count: {ele_num}")
# print(f"Omniact count: {num_omniact}")
# print(f"Mind2web_test count: {num_mind2web_test}")
# print(f"Screenspot count: {num_screenspot}")
# print("Success")


# # GUIEnv: Add bbox for AGUVIS data
# # GUIEnv original data
# guienv_origin_data_1 = json.load(open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/GUIEnv/ocr_grounding_train_stage1_data.json", "r"))
# guienv_origin_data_2 = json.load(open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/GUIEnv/ocr_grounding_train_stage2_data.json", "r"))
# guienv_origin_data = guienv_origin_data_1 + guienv_origin_data_2

# pattern_imgid = re.compile(r'uid_img_(.*?)_(text2bbox|bbox2text)')
# images_id = {}

# pattern_bbox = re.compile(r'<box>(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)</box>')
# for item in tqdm(guienv_origin_data):

#     if item["task_type"] == "bbox2text":
#         continue

#     uid = item["uid"]
#     match = pattern_imgid.search(uid)
#     assert match is not None, f"not match: {uid}"
    
#     img_filename = match.group(1)
#     if img_filename not in images_id:
#         images_id[img_filename] = []
    
#     # Add GUI element information corresponding to this item to the list of the corresponding img_id
#     image_size = item["image_size"]
#     instruction = item["question"]

#     if len(item["answer"]["absolute"]) != 1:
#         continue
#     if len(item["answer"]["related"]) != 1:
#         continue
#     bbox_abs_str = item["answer"]["absolute"][0]
#     bbox_rel_str = item["answer"]["related"][0]

#     match = pattern_bbox.search(bbox_abs_str)
#     assert match is not None, f"not match: {bbox_abs_str}"
#     num1, num2, num3, num4 = match.groups()
#     num1, num2, num3, num4 = float(num1), float(num2), float(num3), float(num4)
#     bbox_abs = [num1, num2, num3, num4]

#     match = pattern_bbox.search(bbox_rel_str)
#     assert match is not None, f"not match: {bbox_rel_str}"
#     num1, num2, num3, num4 = match.groups()
#     num1, num2, num3, num4 = float(num1), float(num2), float(num3), float(num4)
#     bbox_rel = [num1, num2, num3, num4]

#     ele_item = {"image_size": image_size, "instruction": instruction, "bbox_abs": bbox_abs, "bbox_rel": bbox_rel}
#     images_id[img_filename].append(ele_item)

# print(f"unique images_id: {len(set(images_id))}")

# # GUIEnv AGUVIS data
# guienv_aguvis_point = json.load(open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/GUIEnv/guienv.json", "r"))

# # For storing all action types
# action_types = set()
# # Regular expression to extract action type - match text after <image> until the first single quote
# action_pattern = re.compile(r'<image>\n(.*?) \'')
# # Regular expression to extract coordinates from pyautogui call
# coordinate_pattern = re.compile(r'pyautogui\.\w+\(x=(\d+\.\d+), y=(\d+\.\d+)\)')

# guienv_aguvis = []
# match_count = 0
# no_match_count = 0

# for item in tqdm(guienv_aguvis_point):
    
#     # Extract action type, discard drag action type
#     conv_instruction = item['conversations'][0]['value']
#     match = action_pattern.search(conv_instruction)
#     if match:
#         action_type = match.group(1)
#         action_types.add(action_type)
#         if action_type == "Drag to select":
#             continue
#     else:
#         print("Unmatched action type:", conv_instruction)
#         continue

#     assert len(item["conversations"]) == 2

#     # Get image filename (without .jpg extension)
#     img_filename = item["image"][:-4]
#     if img_filename not in images_id:
#         print(f"Image not in original data: {img_filename}")
#         no_match_count += 1
#         continue
    
#     # Extract actual text content from instruction - using a more reliable method
#     # From the first single quote to the last single quote
#     first_quote = conv_instruction.find("'")
#     last_quote = conv_instruction.rfind("'")
    
#     if first_quote == -1 or last_quote == -1 or first_quote == last_quote:
#         print(f"Unable to extract instruction text: {conv_instruction}")
#         no_match_count += 1
#         continue
    
#     instruction_text = conv_instruction[first_quote+1:last_quote]
    
#     # Extract coordinates from pyautogui call
#     coord_match = coordinate_pattern.search(item['conversations'][1]['value'])
#     if not coord_match:
#         print(f"Unable to extract coordinates: {item['conversations'][1]['value']}")
#         no_match_count += 1
#         continue
    
#     click_x = float(coord_match.group(1))
#     click_y = float(coord_match.group(2))
    
#     # Look for matching elements in original data
#     elements = images_id[img_filename]
#     found_match = False
    
#     for element in elements:
#         # Check if instruction text matches
#         if instruction_text == element['instruction']:
#             # Calculate center coordinates of element bbox
#             bbox_rel = element['bbox_rel']
#             center_x = (bbox_rel[0] + bbox_rel[2]) / 2
#             center_y = (bbox_rel[1] + bbox_rel[3]) / 2
            
#             # Check if center coordinates are close to click coordinates (allow some error)
#             if abs(center_x - click_x) < 0.02 and abs(center_y - click_y) < 0.02:
#                 # Match found, add bbox_gt
#                 item['conversations'][1]['bbox_gt'] = bbox_rel
#                 guienv_aguvis.append(item)
#                 found_match = True
#                 match_count += 1
#                 break
    
#     if not found_match:
#         print(f"No matching element found: {img_filename}, {instruction_text}")
#         no_match_count += 1

# print(f"Action types: {action_types}")
# print(f"Successful match count: {match_count}")
# print(f"Unmatched count: {no_match_count}")
# print(f"Total processed count: {len(guienv_aguvis)}")

# # Unify all actions to Click on
# normalized_guienv_aguvis = []
# action_pattern_replace = re.compile(r'<image>\n(.*?) \'')
# pyautogui_pattern = re.compile(r'pyautogui\.(\w+)\(')

# unique_imgs = set()
# random.shuffle(guienv_aguvis)
# for item in tqdm(guienv_aguvis):
#     # Deep copy to prevent modifying original data
#     import copy
#     item_copy = copy.deepcopy(item)
#     unique_imgs.add(item_copy['image'])
    
#     # Check action type in human's value
#     human_value = item_copy['conversations'][0]['value']
#     match = action_pattern_replace.search(human_value)
#     if match:
#         action_type = match.group(1)
#         # Only need to process if it's not "Click on"
#         if action_type != "Click on":
#             # Modify human's value
#             first_quote = human_value.find("'")
#             content = human_value[first_quote:]
#             item_copy['conversations'][0]['value'] = f"<image>\nClick on {content}"
            
#             # Also modify gpt's value
#             gpt_value = item_copy['conversations'][1]['value']
#             match = pyautogui_pattern.search(gpt_value)
#             if match:
#                 # Keep coordinate part
#                 coords = gpt_value[gpt_value.find("("):]
#                 # Replace with click function
#                 item_copy['conversations'][1]['value'] = f"pyautogui.click{coords}"
#     else:
#         print("not match")
#         input()
    
#     normalized_guienv_aguvis.append(item_copy)

# print(f"Sample count after unification: {len(normalized_guienv_aguvis)}")
# print(f"unique_imgs: {len(unique_imgs)}")

# # Save unified data
# json.dump(normalized_guienv_aguvis, open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/GUIEnv/guienv_aguvis_bbox.json", "w"), indent=4)
# print("Save completed: guienv_aguvis_bbox.json")


# # GUIAct
# import pandas as pd
# from io import BytesIO
# import base64
# from PIL import Image

# def read_parquet(path):
#     return pd.read_parquet(path, columns=None)

# def read_image_from_qarquet(cur_df, image_id, b64decode=True):
#     cur_image_str = cur_df.loc[image_id]["base64"]
#     if b64decode:
#         return decode_base64_to_image(cur_image_str)
#     else:
#         return Image.open(BytesIO(cur_image_str)).convert("RGB")

# def decode_base64_to_image(base64_string):
#     return Image.open(BytesIO(base64.b64decode(base64_string))).convert("RGB")

# guiact_data = json.load(open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/GUIAct/web-single_train_data.json", "r"))
# print(f"GUIAct data count: {len(guiact_data)}")

# cur_df = read_parquet("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/GUIAct/web-single_train_images.parquet")
# guiact_img_dir = "/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/GUIAct/web_imgs"
# os.makedirs(guiact_img_dir, exist_ok=True)

# # Regular expression to extract coordinates from box tag
# pattern_bbox = re.compile(r'<box>(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)</box>')

# # Convert GUIAct data to training format
# unique_imgs = set()
# guiact_aguvis = []
# random.shuffle(guiact_data)
# for item in tqdm(guiact_data):
    
#     # 1. Extract basic information
#     image_id = item["image_id"]
#     instruction = item["question"]
    
#     # Check if there are actions_label
#     if not item["actions_label"]:
#         print(f"No action label: {image_id}")
#         continue
    
#     # Only process the first action (click operation)
#     if not len(item["actions_label"]) == 1:
#         print(f"Action count not equal to 1: {image_id}")
#         continue
    
#     action = item["actions_label"][0]
    
#     # Ensure it's a click operation
#     if action["name"] != "click":
#         print(f"Non-click operation: {image_id}, {action['name']}")
#         continue
    
#     # 2. Extract bbox
#     element = action["element"]
#     if "related" not in element:
#         print(f"No relative coordinates: {image_id}")
#         continue
    
#     # 3. Extract normalized coordinates from related
#     related_bbox_str = element["related"]
#     match = pattern_bbox.search(related_bbox_str)
#     if not match:
#         print(f"Coordinate format doesn't match: {related_bbox_str}")
#         continue
    
#     # Extract coordinates
#     num1, num2, num3, num4 = match.groups()
#     bbox_rel = [float(num1), float(num2), float(num3), float(num4)]
    
#     # 4. Calculate center point coordinates
#     center_x = (bbox_rel[0] + bbox_rel[2]) / 2
#     center_y = (bbox_rel[1] + bbox_rel[3]) / 2
    
#     # 5. Create training data format
#     image = read_image_from_qarquet(cur_df, image_id)
#     img_filename = f"{image_id}.png"
#     img_path = os.path.join(guiact_img_dir, img_filename)
#     if not os.path.exists(img_path):
#         image.save(img_path)
    
#     # Construct human instruction
#     human_instruction = f"<image> {instruction}"
    
#     # Construct click operation
#     click_action = f"pyautogui.click(x={center_x:.4f}, y={center_y:.4f})"
    
#     # Create conversation list
#     conversation = [
#         {
#             "from": "human",
#             "value": human_instruction
#         },
#         {
#             "from": "gpt", 
#             "value": click_action,
#             "recipient": "os",
#             "end_turn": True,
#             "bbox_gt": bbox_rel
#         }
#     ]
    
#     # Add to converted dataset
#     data_item = {
#         "image": img_filename,
#         "conversations": conversation
#     }

#     # Check validity
#     if is_bbox_valid(data_item):
#         guiact_aguvis.append(data_item)
#         unique_imgs.add(img_filename)
#     else:
#         print(f"Discarding invalid GUIAct data item: {img_filename}")
    
# print(f"GUIAct sample count after conversion: {len(guiact_aguvis)}")
# print(f"unique_imgs: {len(unique_imgs)}")
# # Save converted data
# json.dump(guiact_aguvis, open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/GUIAct/guiact_aguvis_bbox.json", "w"), indent=4)
# print("Save completed: guiact_aguvis_bbox.json")


# # AndroidControl
# import pickle

# # Get all bounding boxes from interface metadata
# def extract_bbox_from_metadata(metadata_list):

#     valid_bboxes = []

#     def check_bbox_valid(bbox):
#         """Check if the bounding box is valid"""
#         left, top, right, bottom = bbox
#         return left < right and top < bottom

#     def extract_from_list(meta_list):
#         for item in meta_list:
#             if 'bounds_in_screen' in item:
#                 bounds = item['bounds_in_screen']
#                 bbox = [bounds['left'], bounds['top'], bounds['right'], bounds['bottom']]
#                 if check_bbox_valid(bbox) and ('is_visible_to_user' in item) and (item['is_visible_to_user'] == True):
#                     valid_bboxes.append(bbox)
#             if 'tree' in item:
#                 extract_from_list(item['tree'])

#     assert isinstance(metadata_list, list)
#     extract_from_list(metadata_list)

#     return valid_bboxes

# # Extract epoch_id and step_id from filename
# def extract_info_from_filename(filename):
#     # Extract content inside []
#     bracket_match = re.search(r'\[(\d+)\]', filename)
#     bracket_content = bracket_match.group(1) if bracket_match else None
    
#     # Extract content after the last _ and before .pkl
#     last_part_match = re.search(r'_(\d+)\.pkl$', filename)
#     last_part = last_part_match.group(1) if last_part_match else None

#     if last_part is None or bracket_content is None:
#         print(f"Incorrect filename format: {filename}")
#         input()
    
#     return bracket_content, last_part
        
    
# # # Get all .pkl files in directory
# # pkl_dir = '/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/AndroidControl/metadata/all_forest_dict'
# # pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]

# # id_2_bboxes = {}
# # random.shuffle(pkl_files)
# # # Iterate through all pkl files
# # for pkl_file in tqdm(pkl_files):
# #     pkl_path = os.path.join(pkl_dir, pkl_file)

# #     epoch_id, step_id = extract_info_from_filename(pkl_file)
# #     state_id = f"epoch{epoch_id}_step{step_id}"

# #     # Read pickle file
# #     with open(pkl_path, 'rb') as f:
# #         data = pickle.load(f)

# #     bboxes = extract_bbox_from_metadata(data)

# #     if state_id in id_2_bboxes:
# #         print(f"Duplicate state_id: {state_id}")
# #         input()
# #     id_2_bboxes[state_id] = bboxes

# # print(f"id_2_bboxes: {len(id_2_bboxes)}")

# # json.dump(id_2_bboxes, open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/AndroidControl/id_2_bboxes.json", "w"), indent=4)
# # print("Save completed: id_2_bboxes.json")


# id_2_bboxes = json.load(open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/AndroidControl/id_2_bboxes.json", "r"))
# print("load id_2_bboxes done")

# data_split = json.load(open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/AndroidControl/splits.json", "r"))
# print("load data_split done")
# train_episode_ids = data_split['train']
# print(f"train_episode_ids: {len(train_episode_ids)}")

# androidcontrol_data_path = "/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/AndroidControl/parsed_android_control.jsonl"
# imgs_dir = '/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/AndroidControl/tfrecord/images'
# click_num = 0
# correspond_num = 0
# androidcontrol_data_file = []
# unique_imgs = set()
# with open(androidcontrol_data_path, "r") as f:
#     for line in tqdm(f):
#         data = json.loads(line)

#         if data['episode_id'] not in train_episode_ids:
#             continue

#         # Verify data
#         assert len(data['step_instructions']) == len(data['actions'])
#         assert len(data['step_instructions']) == len(data['screenshots_path'])-1
#         for i in range(len(data['step_instructions'])):
#             state_id = f"epoch{data['episode_id']}_step{i}"
#             assert state_id in id_2_bboxes
#         state_id_more = f"epoch{data['episode_id']}_step{len(data['screenshots_path'])}"
#         assert not state_id_more in id_2_bboxes

#         for j, action in enumerate(data['actions']):
#             if action['action_type'] == 'click':
#                 click_num += 1

#                 click_x = action['x']
#                 click_y = action['y']

#                 state_id = f"epoch{data['episode_id']}_step{j}"
#                 bboxes = id_2_bboxes[state_id]
#                 bbox_target = []
#                 for bbox in bboxes:
#                     center_x = (bbox[0] + bbox[2]) / 2
#                     center_y = (bbox[1] + bbox[3]) / 2
                
#                     if (abs(center_x - click_x) <= 1 and abs(center_y - click_y) <= 1):
#                         bbox_target.append(bbox)

#                 # Filter invalid bboxes and select the smallest valid bbox
#                 valid_bbox_target = []
#                 for bbox in bbox_target:
#                     # Check if bbox is valid (width and height > 0)
#                     if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
#                         valid_bbox_target.append(bbox)
                
#                 if len(valid_bbox_target) > 0:
#                     min_area = float('inf')
#                     min_bbox = None
#                     for bbox in valid_bbox_target:
#                         area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
#                         if area < min_area:
#                             min_area = area
#                             min_bbox = bbox
#                     correspond_num += 1
#                 else:
#                     continue

#                 # Build data for each click action with corresponding bbox
#                 instruction = data['step_instructions'][j]
#                 img_filename = data['screenshots_path'][j].split('/')[-1]
#                 img_path = os.path.join(imgs_dir, img_filename)
#                 # if not os.path.exists(img_path):
#                 #     print(f"Image doesn't exist: {img_path}")
#                 #     input()

#                 image = Image.open(img_path)
#                 img_w, img_h = image.size
                
#                 # Normalize bbox and click point
#                 min_bbox = [min_bbox[0]/img_w, min_bbox[1]/img_h, min_bbox[2]/img_w, min_bbox[3]/img_h]
#                 click_x = click_x / img_w
#                 click_y = click_y / img_h

#                 human_instruction = f"<image> {instruction}"
#                 click_action = f"pyautogui.click(x={click_x:.4f}, y={click_y:.4f})"

#                 conversation = [
#                     {"from": "human", "value": human_instruction},
#                     {"from": "gpt", "value": click_action, "recipient": "os", "end_turn": True, "bbox_gt": min_bbox}
#                 ]

#                 data_item = {
#                     "image": img_filename,
#                     "conversations": conversation
#                 }

#                 if is_bbox_valid(data_item):
#                     androidcontrol_data_file.append(data_item)
#                     unique_imgs.add(img_filename)
#                 else:
#                     print(f"Discarding invalid AndroidControl data item: {img_filename}")

#         # if len(androidcontrol_data_file) > 1000:
#         #     break
                
# print(f"click_num: {click_num}")
# print(f"correspond_num: {correspond_num}")
# print(f"unique_imgs: {len(unique_imgs)}")
# print(f"androidcontrol_data_file: {len(androidcontrol_data_file)}")

# json.dump(androidcontrol_data_file, open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/AndroidControl/androidcontrol_aguvis_bbox.json", "w"), indent=4)
# print("Save completed: androidcontrol_aguvis_bbox.json")


# # # Uground
# # import json, glob
# # all_metadata = []
# # for path in tqdm(sorted(glob.glob("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/Uground/uground_metadata_*.json"))):
# #     with open(path) as f:
# #         all_metadata.extend(json.load(f))
# # with open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/Uground/uground_metadata.json", "w") as f:
# #     json.dump(all_metadata, f, indent=4)
# # print("Merge completed: uground_metadata.json")

# uground_data = json.load(open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/Uground/uground_metadata.json", "r"))
# img_dir = "/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/Uground/images"
# print(f"uground_data: {len(uground_data)}")

# uground_data_aguvis = []
# ele_num = 0
# random.shuffle(uground_data)
# for item in tqdm(uground_data):

#     img_filename = item['image']
#     img_path = os.path.join(img_dir, img_filename)
#     # if not os.path.exists(img_path):
#     #     print(f"Image doesn't exist: {img_path}")
#     #     input()

#     conversations = eval(item['conversations'])
#     instruct_2_bbox = []
#     for i in range(int(len(conversations)/2)):
#         instruct = conversations[2*i]["value"]
#         bbox = conversations[2*i+1]["value"]
#         bbox = list(eval(bbox))
#         bbox = [bbox[0]/1000, bbox[1]/1000, bbox[2]/1000, bbox[3]/1000]
#         instruct_2_bbox.append([instruct, bbox])

#     random.shuffle(instruct_2_bbox)
#     item_ele_num = 0
#     conversation = []
#     for i, (instruction, bbox) in enumerate(instruct_2_bbox):
#         if i == 0:
#             instruction_input = f"<image> {instruction}"
#         else:
#             instruction_input = instruction

#         center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
#         x, y = center[0], center[1]
#         action_input = f"pyautogui.click(x={x:.4f}, y={y:.4f})"

#         conversation.append({
#             "from": "human",
#             "value": instruction_input
#         })
#         conversation.append({
#             "from": "gpt",
#             "value": action_input,
#             "recipient": "os",
#             "end_turn": True,
#             "bbox_gt": bbox
#         })
#         item_ele_num += 1

#     data_item = {
#         "image": img_filename,
#         "conversations": conversation
#     }

#     # Use function to check if bbox is valid
#     if is_bbox_valid(data_item):
#         uground_data_aguvis.append(data_item)
#         ele_num += item_ele_num  # Only count elements when data item is valid

# print(f"uground_data_aguvis: {len(uground_data_aguvis)}")
# print(f"ele_num: {ele_num}")
# json.dump(uground_data_aguvis, open("/home/v-kancheng/blob/qianhuiwu_kanzhi/datasets/Uground/uground_aguvis_bbox.json", "w"), indent=4)
# print("Save completed: uground_aguvis_bbox.json")