from PIL import Image, ImageDraw, ImageColor
import json
import os

def dump_args_to_json(model_config, data_processor, model_args, data_args, training_args, output_dir):
    def is_json_serializable(v):
        try:
            json.dumps(v)
            return True
        except:
            return False

    save_path = f"{output_dir}/args.json"
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            json.dump({
                "model_config": {k: v for k, v in model_config.__dict__.items() if is_json_serializable(v)},
                "data_processor_config": {k: v for k, v in data_processor.__dict__.items() if is_json_serializable(v)},
                "image_processor_config": {k: v for k, v in data_processor.image_processor.__dict__.items() if is_json_serializable(v)},
                "model_args": {k: v for k, v in model_args.__dict__.items() if is_json_serializable(v)},
                "data_args": {k: v for k, v in data_args.__dict__.items() if is_json_serializable(v)},
                "training_args": {k: v for k, v in training_args.__dict__.items() if is_json_serializable(v)},
            }, f, indent=4)

def draw_point(image: Image.Image, point: list, color=None):
    if isinstance(color, str):
        try:
            color = ImageColor.getrgb(color)
            color = color + (128,)  
        except ValueError:
            color = (255, 0, 0, 128)  
    else:
        color = (255, 0, 0, 128)  

    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    radius = 14
    x, y = point

    overlay_draw.rectangle(
        [x - radius, y - radius, x + radius, y + radius],
        fill=color
    )
    
    center_radius = radius * 0.1
    overlay_draw.ellipse(
        [(x - center_radius, y - center_radius), 
         (x + center_radius, y + center_radius)],
        fill=(0, 255, 0, 255)
    )

    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)

    return combined.convert('RGB')

def draw_bbox(image: Image.Image, bbox: list, color=None):
    """bbox is in the format of [x1, y1, x2, y2]"""
    if isinstance(color, str):
        try:
            color = ImageColor.getrgb(color)
            color = color + (128,)  
        except ValueError:
            color = (255, 0, 0, 128)  
    else:
        color = (255, 0, 0, 128)
    
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(bbox, fill=color)
    return Image.alpha_composite(image, overlay).convert('RGB')

def do_boxes_overlap(box1, box2):
    """
    Check if two boxes overlap.
    
    Each box is represented as a tuple: (x1, y1, x2, y2)
    Where (x1, y1) is the top-left and (x2, y2) is the bottom-right corner.
    """
    # Unpack the coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Check for no overlap
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False

    return True