import json
import os
from tqdm import tqdm
import numpy as np
from shapely.geometry import Polygon
import cv2
import mmocr.utils as utils
from data_process_on_site import unit_det
import re
import random

text_id = 0
vertical_id = 1
license_pl = 2
directory_path = '/home/ubuntu/workspace/text_det_data/on-site'
threshold = 0.7
split_ratio = 0.95
use_units=False

box_to_polygon = lambda x, y, w, h: [x, y, x + w, y, x + w, y + h, x, y + h]
# Reshape coords to Nx2 array
def convert2polygon(coords):
    coords = np.array(coords).reshape(-1, 2)
    poly = Polygon(coords)
    return poly

def is_bounding_box_larger_than_threshold(bbox, threshold=120):
    """
    Check if the area of a bounding box is larger than a given threshold.

    Parameters:
    - x1, y1: Coordinates of the top-left corner of the bounding box.
    - x2, y2: Coordinates of the bottom-right corner of the bounding box.
    - threshold: The area threshold to compare against.

    Returns:
    - True if the area of the bounding box is larger than the threshold, False otherwise.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height

    return area > threshold


data = {"metainfo": {"dataset_type": "TextDetDataset",
                     "task_name": "textdet",
                     "category": [{"id": 0, "name": "text"}]}
        }

data_list = []
def convert_floats(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_floats(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_floats(value) for key, value in obj.items()}
    return obj

def remove_special_characters(input_str):
    """
    Remove any character that is not A-Z or 0-9 from the string.

    Parameters:
    - input_str: The input string to be processed.

    Returns:
    - A string with all characters not in A-Z or 0-9 removed.
    """
    # Regular expression pattern to match any character that is NOT A-Z, a-z, or 0-9
    pattern = r'[^A-Za-z0-9]'
    # Replace found characters with nothing (i.e., remove them)
    cleaned_str = re.sub(pattern, '', input_str)
    
    return cleaned_str


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
def serialize_json(data):
    """
    Serialize a data structure containing numpy types to a JSON string.

    Parameters:
    - data: The data structure to serialize.

    Returns:
    - A JSON string representation of `data` with numpy types converted to native Python types.
    """
    try:
        json_str = json.dumps(data, cls=NumpyEncoder)
        return json_str
    except TypeError as e:
        return str(e)

def enlarge_polygon(polygon,enlargement_ratio_y=0.0,enlargement_ratio_x=0.0):
    centroid_x = sum(polygon[::2]) / (len(polygon) // 2)
    centroid_y = sum(polygon[1::2]) / (len(polygon) // 2)

    # Enlarge each coordinate by moving it away from the centroid by 10%

    enlarged_polygon = []
    for i in range(0, len(polygon), 2):
        x = polygon[i]
        y = polygon[i + 1]

        # Adjust the x and y coordinates
        x += (x - centroid_x) * enlargement_ratio_x
        y += (y - centroid_y) * enlargement_ratio_y

        enlarged_polygon.extend([x, y])
    return enlarged_polygon

def covnertpoly2bbox(polygon):
    x_coords = polygon[0::2]
    y_coords = polygon[1::2]

    # Find min and max x and y coordinates
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    bbox = [min_x, min_y, max_x, max_y]
    bbox = [round(num, 2) for num in bbox]
    return bbox

items = os.listdir(directory_path)    
num_instances=0

random.shuffle(items)  # Shuffle the list in place
half_index = 2*len(items) // 3  # Calculate the half point (integer division)
items = items[:half_index]  # Return the first half of the list


for item in tqdm(items):
    print(f'folder：{item}')
    src_json_path = os.path.join(directory_path,item,'labels/v1.0.0')
    src_img_path = os.path.join(directory_path,item,'images')
    unit_text_path = os.path.join(directory_path,item,'images_unit_shared_quad_txts')
    relative_img_path = os.path.join(item,'images')
    check_units = True
    if use_units:
        check_units = os.path.exists(unit_text_path)
                    

    if os.path.exists(src_json_path) and os.path.exists(src_img_path) and os.path.exists(unit_text_path)  : 
        for i, filename in enumerate(os.listdir(src_json_path)):
            if filename.endswith('.json'):
                with open(os.path.join(src_json_path, filename)) as f:
                    img_annos = json.load(f)
                txt_file = filename.replace('.json', '.txt')
                png_file = filename.replace('.json', '.png')
                png_file = os.path.join(src_img_path,png_file)
                image_file = ''
                if os.path.exists(png_file):
                    #print(f"File exists: {png_file}")
                    image_file = png_file
                else:
                    jpg_file = filename.replace('.json', '.jpg')
                    jpg_file = os.path.join(src_img_path,jpg_file)
                    if not os.path.exists(jpg_file):
                        print(f"File not exists: {jpg_file}") 
                        continue
                    else:
                        image_file = jpg_file                       
                img_t = cv2.imread(image_file)
                img_height, img_width = img_t.shape[:2]

                bboxes, scores ,texts = unit_det(os.path.join(unit_text_path,txt_file))

                instances = []
                codes = []
                for annot in img_annos:
                    if annot.get("id_type")=="Carrier-ID":
                        continue
                    polygon = annot['polygon']
                    box = covnertpoly2bbox(polygon)
                    bbox_label = 0
                    id = annot['id']
                    if not id:
                        continue

                    polygon = enlarge_polygon(polygon)

                    instance = {"polygon": polygon,
                                "bbox": box,
                                "bbox_label": 0,
                                "text":id,
                                "ignore": False}
                    codes.append(instance)                    
                    instances.append(instance)
                    num_instances+=1
                if use_units:            
                    bboxes, scores ,texts = unit_det(os.path.join(unit_text_path,txt_file))                                
                    for detected_polygon,text,score in zip(bboxes,texts,scores):
                        # Convert detected_polygon to bounding box format if needed
                        # For simplicity, assuming detected_polygon is already in bbox format
                        text = remove_special_characters(text)
                        
                        if not text:
                            continue

                        if score < threshold:                        
                            continue                    
                        detected_polygon = detected_polygon.reshape(-1)
                        gbbox = covnertpoly2bbox(detected_polygon)
                        if not is_bounding_box_larger_than_threshold(gbbox):
                            continue    
                        intersect = False
                        for code in codes:
                            if utils.polygon_utils.poly_intersection(convert2polygon(code['polygon']), convert2polygon(detected_polygon)) != 0.0:
                                intersect = True

                        if not intersect:
                            instance = {"polygon": detected_polygon.tolist(),
                                        "bbox": covnertpoly2bbox(detected_polygon),
                                        "bbox_label": 0,
                                        "text":text,
                                        "ignore": False}
                            instances.append(instance)
                            num_instances+=1
                os.path.join(relative_img_path, os.path.basename(image_file))
                img_data = {"instances": instances,
                            "img_path": os.path.join(relative_img_path, os.path.basename(image_file)),
                            "height": img_height,
                            "width": img_width,
                            "seg_map": ""}
                data_list.append(img_data)                
                #print (data_list)
                

data["data_list"] = data_list
print (f'instances num:{num_instances}')
# Split the data_list into train and test lists

split_index = int(len(data_list) * split_ratio)
train_data_list = data_list[:split_index]
test_data_list = data_list[split_index:]

# Save train data to textdet_train.json
data['data_list'] = train_data_list
data2 =  convert_floats(data)
json_str = serialize_json(data2)

with open('textspotting_train.json', 'w') as file:
    file.write(json_str)

# Save test data to textdet_test.json
data['data_list'] = test_data_list
data2 =  convert_floats(data)
json_str = serialize_json(data2)
with open('textspotting_test.json', 'w') as file:
    file.write(json_str)









