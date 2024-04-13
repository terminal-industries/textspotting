import json
import numpy as np
from find_shortest_edge_and_midpoint import find_shortest_edge_and_midpoint
# Load A.json

data_key='test'
voc_len =96 

with open(f'/home/ubuntu/workspace/projects/str-train/mmocr/textspotting_{data_key}.json') as file:
    a_data = json.load(file)

if voc_len==37:
    CTLABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
            't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '']
else:
    CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/',
                '0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@',
                'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V',
                'W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l',
                'm','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
                            

def simplify_to_rectangle(polygon_coords):
    """
    Simplifies a polygon represented by a list of coordinates to a four-vertex rectangle (bounding box).
    
    Parameters:
    polygon_coords (list): A list of coordinates in the format [x1, y1, x2, y2, ..., xn, yn].
    
    Returns:
    list: Coordinates of the four vertices of the bounding box in the format [x1, y1, x2, y2, x3, y3, x4, y4].
    """
    # Split coordinates into x and y lists
    x_coords = polygon_coords[::2]
    y_coords = polygon_coords[1::2]

    # Determine the bounding box
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Define the four corners of the bounding box
    bounding_box = [min_x, min_y, min_x, max_y, max_x, max_y, max_x, min_y]

    return bounding_box            

def convert_to_wh_format(bbox):
    x1, y1, x2, y2 =bbox
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x1, y1, w, h) format.
    
    Parameters:
    - x1, y1: Coordinates of the top-left corner of the bounding box.
    - x2, y2: Coordinates of the bottom-right corner of the bounding box.
    
    Returns:
    - A tuple containing the coordinates in (x1, y1, w, h) format.
    """
    w = x2 - x1  # Calculate width
    h = y2 - y1  # Calculate height
    
    return [x1, y1, w, h]


# Convert text to indices, pad to length 25
def text_to_indices(text,voc_len=voc_len):
    indices = [CTLABELS.index(c) if c in CTLABELS else voc_len for c in text.lower()][:25]  # Trim/pad to 25
    return indices + [voc_len] * (25 - len(indices))  # Pad with 37/96 if shorter than 25

coco_data = {
    "licenses": [],
    "info": {},
    "categories": [
        {
            "id": 1,
            "name": "text",
            "supercategory": "beverage",
            "keypoints": []
        }
    ],
    "images": [],
    "annotations": []
}


def polygon_to_bezier(polygon):
    """
    Interpolate 8 points of a polygon into 8 Bézier curve control points.
    Input polygon should be a list of four points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    The function returns 8 control points for two Bézier curves:
    - The first curve between P1 and P2
    - The second curve between P3 and P4
    """
    if len(polygon) != 8:
        raise ValueError("Polygon must have exactly four points.")

    points = np.array(polygon).reshape(4, 2)        
    
    # Convert polygon to numpy array for easier manipulation
    # points = np.array(polygon)
    
    # Initialize an array for Bézier control points
    bezier_points = np.zeros((8, 2))
    
    # Insert two control points between P1 and P2
    bezier_points[0] = points[0]  # P1
    bezier_points[1] = points[0] + 1/3 * (points[1] - points[0])  # Control point 1
    bezier_points[2] = points[0] + 2/3 * (points[1] - points[0])  # Control point 2
    bezier_points[3] = points[1]  # P2
    
    # Insert two control points between P3 and P4
    bezier_points[4] = points[2]  # P3
    bezier_points[5] = points[2] + 1/3 * (points[3] - points[2])  # Control point 3
    bezier_points[6] = points[2] + 2/3 * (points[3] - points[2])  # Control point 4
    bezier_points[7] = points[3]  # P4
    
    return bezier_points.flatten().tolist()

anno_id = 0
for data_index, data_item in enumerate(a_data["data_list"]):
    coco_data["images"].append({
        "coco_url": "",
        "date_captured": "",
        "file_name": data_item["img_path"],
        "flickr_url": "",
        "id": data_index,
        "license": 0,
        "width": data_item["width"],
        "height": data_item["height"]
    })
    
    for instance_index, instance in enumerate(data_item["instances"]):
        if len(instance["text"])==0:
            continue
        if len(instance["polygon"])!= 8:
            if len(instance["polygon"])== 10:
                _,instance["polygon"] = find_shortest_edge_and_midpoint(instance["polygon"])
            if len(instance["polygon"])== 12:
                _,polygon = find_shortest_edge_and_midpoint(instance["polygon"])
                _,instance["polygon"] = find_shortest_edge_and_midpoint(polygon)
            if len(instance["polygon"])== 14:
                print (instance["polygon"])
                _,polygon = find_shortest_edge_and_midpoint(instance["polygon"])
                _,polygon = find_shortest_edge_and_midpoint(polygon)
                _,instance["polygon"] = find_shortest_edge_and_midpoint(polygon)
                            
        coco_data["annotations"].append({
            "area": 0,  # Placeholder, should calculate actual area
            "bbox": convert_to_wh_format(instance["bbox"]),
            "category_id": 1,
            "id": anno_id,
            "image_id": data_index,
            "iscrowd": 0,
            "bezier_pts": polygon_to_bezier(instance["polygon"]),
            "rec": text_to_indices(instance["text"])            
        })
        anno_id +=1

# Save to COCO.json
print (f'annot:{len(coco_data["annotations"])},imgs:{len(coco_data["images"])}')

with open(f'text_spotting_{data_key}_{voc_len}_coco.json', 'w') as outfile:
    json.dump(coco_data, outfile, indent=4)

print("Conversion completed and saved as COCO.json")