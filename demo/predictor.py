import numpy as np
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import matplotlib.pyplot as plt

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from adet.utils.visualizer import TextVisualizer
from adet.modeling import swin, vitae_v2
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from adet.data.augmentation import Pad
from scipy.spatial import ConvexHull
from shapely.geometry import LineString
from shapely.geometry import Polygon

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cfg = cfg
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.vis_text = cfg.MODEL.TRANSFORMER.ENABLED
        self.predictor = DefaultPredictor(cfg)
        self.voc_size = 37
        self.CTLABELS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']

    def _process_ctrl_pnt(self, pnt):
        points = pnt.reshape(-1, 2)
        return points

    def _ctc_decode_recognition(self, rec):
        last_char = '###'
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    if self.voc_size == 37 or self.voc_size == 96:
                        s += self.CTLABELS[c]
                        last_char = c
                    else:
                        s += str(chr(self.CTLABELS[c]))
                        last_char = c
            else:
                last_char = '###'
        return s
    def _plot_minimum_covering_polygon(self,bboxes):
        """
        Plots the minimum covering polygon for a list of bounding boxes.
        
        Parameters:
        - bboxes: List of bounding boxes, where each bbox is [min_x, min_y, max_x, max_y]
        """
        polygon = []

        for bbox in bboxes:
            min_x, min_y = bbox[0], bbox[1]
            max_x, max_y = bbox[2], bbox[3]
            polygon.append((min_x, min_y))
            polygon.append((max_x, max_y))

        points = np.array(polygon)
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_polygon = Polygon(hull_points)
        min_rectangle = hull_polygon.minimum_rotated_rectangle
        return min_rectangle

    def run_on_image(self, image,visulize_on_img=False):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = TextVisualizer(image, self.metadata, instance_mode=self.instance_mode, cfg=self.cfg)
        instances = predictions["instances"].to(self.cpu_device)
        results = self.convert_prediction(instances)
        return results

    def convert_prediction(self,predictions):
        
        ctrl_pnts = predictions.ctrl_points.numpy()
        scores = predictions.scores.tolist()
        recs = predictions.recs
        bd_pts = np.asarray(predictions.bd)   
        results = []
        for ctrl_pnt, score, rec, bd in zip(ctrl_pnts, scores, recs, bd_pts):
            # draw polygons
            if bd is not None:                
                polygon =self._plot_minimum_covering_polygon(bd)

            # draw center lines
            text = self._ctc_decode_recognition(rec)
            if self.voc_size == 37:
                text = text.upper()
            # text = "{:.2f}: {}".format(score, text)
            text = "{}".format(text)
            res = {'polygon':[(round(x, 2), round(y, 2)) for x, y in polygon.exterior.coords],'text':text}
            results.append(res)

        return results            



    def _ctc_decode_recognition(self, rec):
        last_char = '###'
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    if self.voc_size == 37 or self.voc_size == 96:
                        s += self.CTLABELS[c]
                        last_char = c
                    else:
                        s += str(chr(self.CTLABELS[c]))
                        last_char = c
            else:
                last_char = '###'
        return s        