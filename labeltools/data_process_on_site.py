import numpy as np
import cv2
from shapely.geometry import Polygon
from PIL import Image, ImageDraw, ImageFont
import os

"""
process data to deepSolo Format
"""


def compute_iou(poly1, poly2):
    """计算两个多边形的IoU。"""
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        iou = inter_area / union_area
        return iou
    except:
        return 0

def nms_polygons(polygons, scores, iou_threshold=0.5):
    """基于多边形的NMS实现。
    :param polygons: 多边形列表，每个多边形由其顶点坐标定义。
    :param scores: 对应多边形的置信度列表。
    :param iou_threshold: IoU阈值用于决定是否抑制。
    :return: 保留的多边形索引。
    """
    # 首先将所有多边形坐标转换为Polygon对象
    poly_objects = [Polygon(poly) for poly in polygons]

    # 根据分数降序排列多边形
    indices = list(range(len(poly_objects)))
    indices.sort(key=lambda i: scores[i], reverse=True)

    keep = []
    while indices:
        current = indices.pop(0)
        keep.append(current)
        for future in list(indices):
            iou = compute_iou(poly_objects[current], poly_objects[future])
            if iou > iou_threshold:
                indices.remove(future)
    return keep

def display_on_img(polygons, scores, texts,image):
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font_size = 12  # 设置较大的字体大小
    font = ImageFont.truetype(font_path, font_size)

    for polygon, score, text in zip(polygons, scores, texts):
        # 绘制多边形
        
        coords_pillow = list(map(tuple, polygon))
        draw.polygon(coords_pillow, outline="red")
        
        # 计算多边形的边界框，用于放置文本
        x_min = min([point[0] for point in polygon])
        y_min = min([point[1] for point in polygon])
        
        # 绘制文本（分数和文本内容）
        draw.text((x_min, y_min), f"{text}", fill="blue", font=font)

    # 保存图片到文件
    return image
    


# 定义一个函数来转换坐标和提取信息
def process_data(data):
    boxes = []  # 存储边界框
    scores = []  # 存储概率（置信度）
    texts = []  # 存储文本内容
    for line in data.split('\n'):
        if not line.strip():
            continue
        parts = line.split('####')
        all_coords = parts[0].split(',')[:8]
        
        coords = list(map(int, all_coords))
        text, score = parts[1].rsplit('_', 1)
        score = float(score.split('=')[1])
        
        # 转换坐标到矩形（x_min, y_min, x_max, y_max）
        #x_min, y_min = min(coords[::2]), min(coords[1::2])
        #x_max, y_max = max(coords[::2]), max(coords[1::2])
        
        boxes.append([(coords[0],coords[1]),(coords[2],coords[3]),(coords[4],coords[5]),(coords[6],coords[7])])
        scores.append(score)
        texts.append(text)
    return np.array(boxes), np.array(scores), texts

# 数据字符串

def unit_det(txt_path):
    with open(txt_path, 'r') as file:    
        data = file.read()
    polygons, scores, texts = process_data(data)
    keep_indices = nms_polygons(polygons, scores, iou_threshold=0.3)
    keep_polygons=[]
    keep_scores=[]
    keep_texts =[]
    # 打印NMS后保留的文本和概率
    for i in keep_indices:
        #i = i[0]  # NMSBoxes返回的是一个二维数组
        #print(f'Text: {texts[i]}, Probability: {scores[i]},coords: {polygons[i]}')
        keep_polygons.append(polygons[i])
        keep_scores.append(scores[i])
        keep_texts.append(texts[i])
    return keep_polygons,keep_scores,keep_texts      


'''
if __name__ == '__main__':
    basepath = '/home/ubuntu/workspace/text_det_data/on-site/2024-02-12'
    image_base = 'images'
    label_base = 'images_unit_shared_quad_txts'
    units_file = 'entry_feb_12_2024_09_10_03_pm_feb_12_2024_09_14_45_pm_fps_5_frame_1386.txt'
    png_file = units_file.split('.')[0]+'.png'
    # 处理数据

    keep_polygons, keep_scores, keep_texts = unit_det(os.path.join(basepath,label_base,units_file))

    image = Image.open(os.path.join(basepath,image_base,png_file))
    pil_img = display_on_img(keep_polygons, keep_scores, keep_texts,image)

    pil_img.save('tmp.jpg')
'''