import json
import numpy as np
from find_shortest_edge_and_midpoint import find_shortest_edge_and_midpoint
import math
from polygon_util.py import order_points_quadrangle,sort_vertical_points
# Load A.json

data_key='train'
#voc_len =96 
voc_len =37
with open(f'textspotting_{data_key}.json') as file:
#with open('textspotting_train_sample.json') as file:
    a_data = json.load(file)
anno_id = 0
if voc_len==37:
    CTLABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
            't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '']
else:
    CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/',
                '0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@',
                'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V',
                'W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l',
                'm','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
                            
def find_top_left_point(points):
    # Sort points by y-value (ascending); if tied, by x-value (ascending)
    sorted_points = sorted(points, key=lambda x: (x[1], x[0]))
    return sorted_points[0]

def sort_polygon_points(width, height, input_points, is_vertical):
    # Define corner points based on image width and height
    # Step 1: Find the top left point (P0)
    points = np.array(input_points).reshape(4, 2) 

    if not is_vertical:
        ordered_points = order_points_quadrangle(points)
    else:

        ordered_points = sort_vertical_points(points)

    return ordered_points.flatten().tolist() 



'''
def find_top_left_point(points):
    # Sort points by y-value (ascending); if tied, by x-value (ascending)
    sorted_points = sorted(points, key=lambda x: (x[1], x[0]))
    return sorted_points[0]

def sort_bezier_points(bezier_pts, is_vertical):
    points = np.array(bezier_pts).reshape(8, 2)

    P0 = find_top_left_point(points)
    
    # Remove P0 from the list of points for further processing
    remaining_points = np.array([p for p in points if not np.array_equal(p, P0)])
    
    if is_vertical:
        # Sort remaining points based on proximity in the Y direction from P0
        sorted_by_y = sorted(remaining_points, key=lambda p: np.abs(p[1] - P0[1]))
        P1, P2, P3 = sorted_by_y[:3]
        
        # Find P7 as the closest point in the X direction from P0
        sorted_by_x = sorted(remaining_points, key=lambda p: np.abs(p[0] - P0[0]))
        P7 = sorted_by_x[0]
        
        # Remove P7 and sort remaining points for P7 in the Y direction
        remaining_for_p7 = [p for p in sorted_by_x[1:] if not np.array_equal(p, P7)]
        sorted_by_y_p7 = sorted(remaining_for_p7, key=lambda p: np.abs(p[1] - P7[1]))
        P6, P5, P4 = sorted_by_y_p7[:3]
    else:
        # Sort remaining points based on proximity in the X direction from P0
        sorted_by_x = sorted(remaining_points, key=lambda p: np.abs(p[0] - P0[0]))
        P1, P2, P3 = sorted_by_x[:3]
        
        # Find P7 as the closest point in the Y direction from P0
        sorted_by_y = sorted(remaining_points, key=lambda p: np.abs(p[1] - P0[1]))
        P7 = sorted_by_y[0]
        
        # Remove P7 and sort remaining points for P7 in the X direction
        remaining_for_p7 = [p for p in sorted_by_y[1:] if not np.array_equal(p, P7)]
        sorted_by_x_p7 = sorted(remaining_for_p7, key=lambda p: np.abs(p[0] - P7[0]))
        P6, P5, P4 = sorted_by_x_p7[:3]
    
    # Assemble all points in the correct order
    reordered_points = np.array([P0, P1, P2, P3, P4, P5, P6, P7])

    return reordered_points.flatten().tolist()
'''    

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


import numpy as np

def polygon_to_bezier(polygon):
    """
    Interpolate Bézier curve control points on the longest side and its opposite side of a quadrilateral.
    Input polygon should be a list of four points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    The function returns 8 control points for two Bézier curves:
    - The first curve on the longest side
    - The second curve on the opposite side of the longest side
    """
    if len(polygon) != 8:
        raise ValueError("Polygon must have exactly four points.")

    vertices = np.array(polygon).reshape(4, 2)
    vertices_list = [tuple(vertex) for vertex in vertices]

    # 计算边的长度和对应的两个顶点
    num_vertices = len(vertices_list)
    edges = [(np.linalg.norm(vertices[i] - vertices[(i + 1) % num_vertices]), i, (i + 1) % num_vertices)
             for i in range(num_vertices)]

    # 按长度排序并取最长的两条边
    longest_edges = sorted(edges, key=lambda x: x[0], reverse=True)[:2]

    # 新的顶点集合，初始包含所有原始顶点
    new_vertices = vertices_list.copy()

    # 处理每一条最长的边，插入两个点
    for _, start, end in longest_edges:
        # 计算插入点的坐标
        p1 = tuple(vertices[start] + (vertices[end] - vertices[start]) * 1 / 3)
        p2 = tuple(vertices[start] + (vertices[end] - vertices[start]) * 2 / 3)
        
        # 在结束点前插入新点
        end_index = new_vertices.index(vertices_list[end])
        new_vertices.insert(end_index, p2)
        new_vertices.insert(end_index, p1)    

    vertices = [round(num, 2) for point in new_vertices for num in point]
    return vertices



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

        #polygon = sort_polygon_points(instance["polygon"],instance['vertical_text'])        
        #btz_points = polygon_to_bezier(instance["polygon"])        
        #btz_points = sort_bezier_points(btz_points, instance['vertical_text'])
        btz_points = sort_polygon_points(data_item["width"],data_item["height"],instance["polygon"],instance['vertical_text'])
        btz_points = polygon_to_bezier(btz_points) 
        coco_data["annotations"].append({
            "area": 0,  # Placeholder, should calculate actual area
            "bbox": convert_to_wh_format(instance["bbox"]),
            "category_id": 1,
            "id": anno_id,
            "image_id": data_index,
            "iscrowd": 0,
            "bezier_pts": btz_points,
            "rec": text_to_indices(instance["text"])            
        })
        anno_id +=1

# Save to COCO.json
print (f'annot:{len(coco_data["annotations"])},imgs:{len(coco_data["images"])}')

with open(f'text_spotting_{data_key}_{voc_len}_coco.json', 'w') as outfile:
    json.dump(coco_data, outfile, indent=4)

print(f"Conversion completed and saved as text_spotting_{data_key}_{voc_len}_coco.json")