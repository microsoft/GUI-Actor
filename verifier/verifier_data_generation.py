import json
import os
import re
import numpy as np
import random
from PIL import Image, ImageDraw
import argparse


dic = {
        "from": "gpt",
        "value": "True",
        "recipient": "os",
        "end_turn": True
    }
neg_dic = {
        "from": "gpt",
        "value": "False",
        "recipient": "os",
        "end_turn": True
    }



def sample_point(bbox):
    x0, y0, x1, y1 = bbox
    t = 0
    while t <= 50:
        xx, yy = np.random.random(2)
        t += 1
        if not ((x0 < xx < x1) and (y0 < yy < y1)):
            break
    if t > 50:
        return
    return xx, yy


def load_json_file(file_path):
    """Load and parse JSON data from a file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' contains invalid JSON.")
        return None


def draw_annotations(img, point_in_pixel, bbox, output_path='test.png', color='red', size=1):
    draw = ImageDraw.Draw(img)

    # Draw the ground truth bounding box in green
    if bbox:
        # Assuming bbox format is [x1, y1, x2, y2]
        draw.rectangle(bbox, outline="yellow", width=4)
    # Draw a small rectangle around the predicted point in red
    if point_in_pixel:
        # Create a small rectangle around the point (5 pixels in each direction)
        radius = np.ceil(8 * size).astype(int)
        circle_bbox = [
            point_in_pixel[0] - radius,  # x1
            point_in_pixel[1] - radius,  # y1
            point_in_pixel[0] + radius,  # x2
            point_in_pixel[1] + radius   # y2
        ]
        draw.ellipse(circle_bbox, outline=color, width=np.ceil(4 * size).astype(int))

    img.save(output_path)
    print(f"Annotated image saved to {output_path}")
    return img


def transform_to_conversation_format(data, file, image_folder_dict, new_directory):
    """
    Transform the input data to the specified conversation format.
    Args:
        data: List of dictionaries containing webpage elements data
    
    Returns:
        List of dictionaries in the conversation format
    """
    image_folder = image_folder_dict[file]
    result = []
    for i, item in enumerate(data):
        print(i / len(data))
        img_filename = item['img_filename']

        prompt = 'Please observe the screenshot and exame whether the hollow red circle accurately placed on the intended position in the image:'
    
        if 'elements' in item:
            # sample n//2 element
            n = len(item['elements'])
            ind_list = []
            if n <= 1:
                ind_list = [0]
            else:
                ind_list = random.sample(range(n), min(n//2, 3))

            for ind in ind_list:
                conversations = []
                instruction = item['elements'][ind]['instruction']
                bbox = item['elements'][ind]['bbox']
                if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= 0.8:
                    continue

                conversations.append({
                    "from": "human",
                    "value": f"<image>\n{prompt} " + f"'{instruction}'. Answer True or False."
                })
                
                # Calculate the center point of the bounding box
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2

                if n >= 2:
                    neg_ind = random.choice([k for k in range(n) if k != ind])
                    neg_bbox = item['elements'][neg_ind]['bbox']
                    x_neg, y_neg =  (neg_bbox[0] + neg_bbox[2]) / 2, (neg_bbox[1] + neg_bbox[3]) / 2
                    if (x_center - x_neg) ** 2 + (y_center - y_neg) ** 2 < 0.05:
                        x_neg, y_neg = sample_point(bbox)
                else:
                    x_neg, y_neg = sample_point(bbox)
            
                # draw image
                try:
                    img = Image.open(os.path.join(image_folder, img_filename))
                except:
                    continue

                
                prefix, suffix = img_filename.split('.')
                width, height = img.size
                save_path = os.path.join(new_directory, file+'_'+ prefix.replace('/', '') + f'_pos{ind}.' + suffix)
                while os.path.exists(save_path):
                    save_path = os.path.join(new_directory, file+'_'+ prefix.replace('/', '') + f'_pos{ind}_{random.randint(0, 1000)}.' + suffix)
            
                try:
                    draw_annotations(img, [x_center * width, y_center* height], None, output_path=save_path, size=height/1000 * 1.2)
                except:
                    continue
                img = Image.open(os.path.join(image_folder, img_filename))
                neg_save_path = os.path.join(new_directory, file+'_'+ prefix.replace('/', '') + f'_neg{ind}.' + suffix)
                while os.path.exists(neg_save_path):
                    neg_save_path = os.path.join(new_directory, file+'_'+ prefix.replace('/', '') + f'_neg{ind}_{random.randint(0, 1000)}.' + suffix)
            
                draw_annotations(img, [x_neg * width, y_neg* height], None, output_path=neg_save_path, size=height/1000 * 1.2)


                # Create the conversation item
                result.append({
                    "image":save_path.replace(new_directory, ''),
                    "conversations": conversations + [dic]
                })
                result.append({
                    "image":neg_save_path.replace(new_directory, ''),
                    "conversations": conversations + [neg_dic]
                })
        else:
            conversations = []
            instruction = item['instruction']
            bbox = item['bbox']
            conversations.append({
                "from": "human",
                "value": f"<image>\n{prompt} " + f"'{instruction}'. Answer True or False."
            })


            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= 0.8:
                continue

            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            x_neg, y_neg = sample_point(bbox)

            # draw image
            try:
                img = Image.open(os.path.join(image_folder, img_filename))
            except:
                continue
            prefix, suffix = img_filename.split('.')
            width, height = img.size
            save_path = os.path.join(new_directory, file+'_'+ prefix.replace('/', '') + '_pos.' + suffix)
            while os.path.exists(save_path):
                save_path = os.path.join(new_directory, file+'_'+ prefix.replace('/', '') + f'_pos_{random.randint(0, 1000)}.' + suffix)
            
            draw_annotations(img, [x_center * width, y_center* height], None, output_path=save_path, size=height/1000 * 1.2)

            img = Image.open(os.path.join(image_folder, img_filename))
            neg_save_path = os.path.join(new_directory, file+'_'+ prefix.replace('/', '') + '_neg.' + suffix)
            while os.path.exists(neg_save_path):
                neg_save_path = os.path.join(new_directory, file+'_'+ prefix.replace('/', '') + f'_neg_{random.randint(0, 1000)}.' + suffix)
            
            draw_annotations(img, [x_neg * width, y_neg* height], None, output_path=neg_save_path, size=height/1000 * 1.2)

            # Create the conversation item
            result.append({
                "image":save_path.replace(new_directory, ''),
                "conversations": conversations + [dic]
            })
            result.append({
                "image":neg_save_path.replace(new_directory, ''),
                "conversations": conversations + [neg_dic]
            })
    
    return result




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate verifier data")
    parser.add_argument('--root_path', type=str, required=True, help='Root path to OS-Atlas-data')
    parser.add_argument('--new_directory', type=str, default='./verifier_data', help='Directory to save the new verifier data')
    parser.add_argument('--file_dict_key', type=str, default='', help='Key for the file dictionary to process')
    parser.add_argument('--save_suffix', type=str, default='verifier', help='Suffix for the saved files')
    parser.add_argument('--selected_size', type=int, default=10000, help='Number of samples to select from each file')
    args = parser.parse_args()


    root_path = args.root_path
    new_directory = args.new_directory
    save_suffix = args.save_suffix
    selected_size = args.selected_size
    
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)


    image_folder_dict = {
        'windows_splited': f'{root_path}/desktop_domain/windows_images',
        'linux_splited': f'{root_path}/desktop_domain/linux_images',
        'macos_splited': f'{root_path}/desktop_domain/macos_images',
        'widget_captioning': f'{root_path}/mobile_domain/combined',
        'uibert_raw': f'{root_path}/mobile_domain/UIBert',
        'ricosca': f'{root_path}/mobile_domain/combined',
        'amex_raw': f'{root_path}/mobile_domain/amex_images',
        'seeclick_web': f'{root_path}/web_domain/seeclick_web_imgs',
        'fineweb_3m': f'{root_path}/web_domain/fineweb'
    }


    file_dict = {
        'desktop_domain': ['linux_splited', 'windows_splited', 'macos_splited'],
        'mobile_domain': ['uibert_raw', 'ricosca', 'amex_raw', 'widget_captioning'],
        'web_domain': ['fineweb_3m', 'seeclick_web'],
    }


    def process_files(directory):
        files = file_dict[directory]
        for file in files:
            file_path = os.path.join(root_path, directory, file + '.json')
            # Load the JSON data
            data = load_json_file(file_path)
            data = random.sample(data, selected_size) if len(data) >= selected_size else data
            print(directory, file, len(data))

            # Extract coordinates
            new_data = transform_to_conversation_format(data, file, image_folder_dict, new_directory)


            print(directory, file, len(data))
            with open(file_path.replace('.json', f'_{save_suffix}.json'), "w", encoding="utf-8") as f:
                json.dump(new_data, f)


    if len(args.file_dict_key) == 0:
        for directory in file_dict.keys():
            process_files(directory)
    else:
        key = args.file_dict_key
        assert key in file_dict.keys(), f"Key {key} not found in file_dict"
        process_files(key)








