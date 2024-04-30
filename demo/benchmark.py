# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import json
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg
from PIL import Image
import numpy as np
import glob
# constants
#  lable /home/ubuntu/workspace/data/benchmark3.5/labels/

def read_gt_info(filename):
    # Strip the ".json" extension from the filename
    with open(filename, 'r') as file:
        # 读取并转换JSON数据到Python数据结构
        json_data = json.load(file)

    base_filename = os.path.splitext(os.path.basename(filename))[0]
    
    # Initialize a list to hold the ID texts
    ids = []
    
    # Loop through each item in the JSON data
    for item in json_data:
        if item['id']:  # Check if the 'id' field is not empty
            ids.append(item['id'].lower())
    
    # Return the result in the specified format
    return f"'{base_filename}'", ids,json_data

def extract_pred(fn,data):
    base_filename = os.path.splitext(os.path.basename(fn))[0]
    texts = []  # 创建一个空列表用来存放文本
    for item in data:  # 遍历列表中的每个字典
        if 'text' in item:  # 检查每个字典是否有'text'这个键
            texts.append(item['text'].lower())  # 如果有，把它的值添加到texts列表中
    return f"'{base_filename}'",texts


def calculate_metrics(detected_texts, ground_truths):
    """
    Calculate precision, recall, F1-score, and accuracy for text detection.

    Parameters:
    - detected_texts: Dictionary of detected texts, where keys are filenames and values are lists of detected strings.
    - ground_truths: Dictionary of ground truth texts, where keys are filenames and values are lists of actual strings.

    Returns:
    - Dictionary containing precision, recall, F1-score, and accuracy.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Loop through each file in the ground truths
    for filename, truths in ground_truths.items():
        detected = set(detected_texts.get(filename, []))
        truth_set = set(truths)
        
        # True positives: Detected texts that are indeed in the ground truth
        true_positives += len(detected.intersection(truth_set))
        
        # False positives: Detected texts that are not in the ground truth
        false_positives += len(detected - truth_set)
        
        # False negatives: Ground truth texts that were not detected
        false_negatives += len(truth_set - detected)
    
    # Calculate precision, recall, and F1-score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate accuracy: (true positives) / (true positives + false positives + false negatives)
    accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
    
    return {
        'precision': "{:.4f}".format(precision),
        'recall': "{:.4f}".format(recall),
        'F1-score': "{:.4f}".format(f1_score),
    }




def find_files(json_dir, image_dir):
    """
    在指定目录中查找所有JSON文件，并在另一个目录中查找对应的图像文件。

    参数:
    - json_dir: 包含JSON文件的目录路径。
    - image_dir: 包含图像文件的目录路径。

    返回:
    - json_files: 找到的JSON文件列表。
    - image_files: 找到的对应的图像文件列表。
    """
    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    image_files = []

    # 图像文件可能的扩展名
    image_extensions = ['*.png']

    # 对于每个JSON文件，找到对应的图像文件
    for json_file in json_files:
        # 去掉扩展名获取基本文件名
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        
        # 在图像目录中搜索同名的图像文件
        #for ext in image_extensions:
        ext = '.png'
        found_images = glob.glob(os.path.join(image_dir, f'{base_name}{ext}'))
        image_files.extend(found_images)

    return json_files, image_files

def draw_polygon_and_text(image, polygon, text):
    pts = np.array(polygon, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=(255, 255, 0), thickness=5)
    cv2.putText(image, text, (int(polygon[0][0]), int(polygon[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    return image                



def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Text spotting benchmark")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images/lables")      
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )  
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def analyze_data(data):
    id_type_counts = {}
    vertical_text_counts = {'true': 0, 'false': 0}

    for item in data:
        # 统计 id_type 出现的次数
        id_type = item.get('id_type', 'Unknown')
        if id_type in id_type_counts:
            id_type_counts[id_type] += 1
        else:
            id_type_counts[id_type] = 1

        # 统计 vertical_text 为 true 和 false 的次数
        vertical_text = item.get('vertical_text', False)
        if vertical_text:
            vertical_text_counts['true'] += 1
        else:
            vertical_text_counts['false'] += 1

    return id_type_counts, vertical_text_counts


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    print(args)
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    
    imgs_path = os.path.join(args.input[0],'images')
    labels_path = os.path.join(args.input[0],'labels')
    anno_list,imgs_list = find_files(labels_path, imgs_path)    
    assert len(anno_list)==len(imgs_list),'labels and images are not same length'
    gts = {}
    preds = {}
    idx = 0
    anno_infos=[]
    for path,anno in tqdm.tqdm(zip(imgs_list, anno_list)):
        idx +=1
        fn,id_info,anno_info = read_gt_info(anno)        
        gts[fn]=id_info
        img = cv2.imread(path)
        start_time = time.time()
        prediction = demo.run_on_image(img)
        fn2,pred = extract_pred(path,prediction)
        preds[fn2]=pred
        anno_infos +=anno_info
    print(calculate_metrics(preds, gts))
    print(analyze_data(anno_infos))
