import cv2
import torch
import time
from instances import Instances
import tensorrt as trt
#from trt_inf_v10 import TRTWrapper as trt_wrapper
from trt_wrapper import TRTWrapper as trt_wrapper
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

INFERENCE_TH_TEST = 0.38

class TxtSpottingImageProcessor:
    def __init__(self, model_path, dims, loop_count=1):
        """
        Initialize the TxtSpottingImageProcessor class.

        Args:
            model_path (str): Path to the TensorRT model file.
            dims (tuple): Dimensions for resizing input images (width, height).
            loop_count (int): Number of loops for inference (default is 1).
        """
        self.infer_th_test = INFERENCE_TH_TEST
        self.model_path = model_path
        self.dims = dims
        self.loop_count = loop_count
        self.sum_time_preprocess = 0
        self.sum_time_network = 0
        self.sum_time_postprocess = 0

        # Initialize TensorRT model
        trt.init_libnvinfer_plugins(None, "")
        self.model = trt_wrapper(self.model_path)        
        self.voc_size = 37
        self.CTLABELS = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 
            't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]

    def postprocess(self, ctrl_point_cls, ctrl_point_coord, ctrl_point_text, bd_points, image_sizes, output_height, output_width, min_size=None, max_size=None):
        """
        Post-process the inference results.

        Args:
            ctrl_point_cls (torch.Tensor): Control point classification tensor.
            ctrl_point_coord (torch.Tensor): Control point coordinates tensor.
            ctrl_point_text (torch.Tensor): Control point text tensor.
            bd_points (torch.Tensor): Bezier points tensor.
            image_sizes (list): List of image sizes.
            output_height (int): Height of the output image.
            output_width (int): Width of the output image.
            min_size (int, optional): Minimum size for scaling.
            max_size (int, optional): Maximum size for scaling.

        Returns:
            list: List of processed results as Instances objects.
        """
        assert ctrl_point_cls.shape[0] == len(image_sizes)
        results = []
        
        # Apply softmax to the control point text and sigmoid to the control point classes
        ctrl_point_text = torch.softmax(ctrl_point_text, dim=-1)
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)

        # Process each image in the batch
        for scores_per_image, labels_per_image, ctrl_point_per_image, ctrl_point_text_per_image, bd, image_size in zip(
            scores, labels, ctrl_point_coord, ctrl_point_text, bd_points, image_sizes):
            selector = scores_per_image >= self.infer_th_test
            scores_per_image = scores_per_image[selector]
            labels_per_image = labels_per_image[selector]
            ctrl_point_per_image = ctrl_point_per_image[selector]
            ctrl_point_text_per_image = ctrl_point_text_per_image[selector]
            bd = bd[selector]

            result = Instances(image_size)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            result.rec_scores = ctrl_point_text_per_image
            ctrl_point_per_image[..., 0] *= image_size[1]
            ctrl_point_per_image[..., 1] *= image_size[0]
            result.ctrl_points = ctrl_point_per_image.flatten(1)
            _, text_pred = ctrl_point_text_per_image.topk(1)
            result.recs = text_pred.squeeze(-1)
            bd[..., 0::2] *= image_size[1]
            bd[..., 1::2] *= image_size[0]
            result.bd = bd
            results.append(result)

        # Scale results to the output size
        if min_size and max_size:
            size = min_size * 1.0
            scale_img_size = min_size / min(output_width, output_height)
            if output_height < output_width:
                newh, neww = size, scale_img_size * output_width
            else:
                newh, neww = scale_img_size * output_height, size
            if max(newh, neww) > max_size:
                scale = max_size * 1.0 / max(newh, neww)
                newh = newh * scale
                neww = neww * scale
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)
            scale_x, scale_y = (output_width / neww, output_height / newh)
        else:
            scale_x, scale_y = (output_width / results[0].image_size[1], output_height / results[0].image_size[0])

        for result in results:
            if result.has("ctrl_points"):
                ctrl_points = result.ctrl_points
                ctrl_points[:, 0::2] *= scale_x
                ctrl_points[:, 1::2] *= scale_y

            if result.has("bd") and not isinstance(result.bd, list):
                bd = result.bd
                bd[..., 0::2] *= scale_x
                bd[..., 1::2] *= scale_y

        return results
    
    def preprocess(self,original_image):        
        """
        Preprocess the input image for model inference.

        Args:
            original_image: The original input cv2 image in numpy array format.

        Returns:
            img_t: Preprocessed image tensor ready for model inference.
        """

        image = cv2.resize(original_image, self.dims)
        img_t = image.transpose(2, 0, 1)        
        #img_t = torch.from_numpy(img_t).unsqueeze(0).float().cuda()
        img_t = torch.from_numpy(img_t).float().cuda()
        return img_t
    

    def benchmark_inference(self,image_path, benchmark_mode=False,iteration=-1, start_events=None, end_events=None):
        """
        Perform model inference speed benchmark.

        Args:
            image_path: Path to the input image.
            benchmark_mode: Boolean flag to enable benchmarking.
            iteration: Current iteration index for benchmarking.
            start_events: List of CUDA events to mark the start of benchmarking.
            end_events: List of CUDA events to mark the end of benchmarking.

        Returns:
            output: The output of the model inference.
        """        

        original_image = cv2.imread(image_path)                
        img_t = self.preprocess(original_image)

        if benchmark_mode:
            start_events[iteration].record()
        output = self.model.forward(dict(image=img_t))
        if benchmark_mode:
            end_events[iteration].record()

        

    def process_and_infer(self, image_path):
        """
        Process the input image and perform inference.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: Processed results and the original image.
        """
        # Load and preprocess the image
        original_image = cv2.imread(image_path)
        t0 = time.time()
        height, width = original_image.shape[:2]
        img_t = self.preprocess(original_image)
        self.sum_time_preprocess += (time.time()-t0)

        # Perform inference
        
        
        t1 = time.time()
        predictions = self.model.forward(dict(image=img_t))
        ctrl_point_cls = predictions['ctrl_point_cls']
        ctrl_point_coord = predictions['ctrl_point_coord']
        ctrl_point_text = predictions['ctrl_point_text']
        bd_points = predictions['bd_pointsoutput']        
        #ctrl_point_cls = torch.from_numpy(predictions[0])
        #ctrl_point_coord = torch.from_numpy(predictions[1])
        #ctrl_point_text = torch.from_numpy(predictions[2])
        #bd_points = torch.from_numpy(predictions[3])

        self.sum_time_network += (time.time()-t1)

        t2 = time.time()
        # Post-process the results
        results = self.postprocess(
            ctrl_point_cls,
            ctrl_point_coord,
            ctrl_point_text,
            bd_points,
            [self.dims],
            height,
            width
        )
        self.sum_time_postprocess += (time.time() - t2)
        processed_results = [{"instances": r} for r in results]
        return processed_results, original_image
    
    def convert_prediction(self, predictions):
        """
        Convert the prediction results to a more readable format.

        Args:
            predictions (Instances): The prediction results as Instances objects.

        Returns:
            list: List of dictionaries containing polygons and recognized text.
            Examples: 
            [{'polygon': [(2847.97, 1391.61), (2665.7, 1396.8), (2663.06, 1303.97), (2845.32, 1298.78), (2847.97, 1391.61)], 'text': '9G79992'}, 
             {'polygon': [(189.44, 601.65), (188.74, 572.49), (243.64, 571.19), (244.33, 600.34), (189.44, 601.65)], 'text': '9G41653'}, 
             {'polygon': [(1261.59, 617.22), (1330.63, 627.07), (1326.37, 657.0), (1257.33, 647.16), (1261.59, 617.22)], 'text': 'FLXZ402068'}]                 
        """
        ctrl_pnts = predictions.ctrl_points.numpy()
        scores = predictions.scores.tolist()
        recs = predictions.recs
        bd_pts = np.asarray(predictions.bd)   
        results = []
        
        for ctrl_pnt, score, rec, bd in zip(ctrl_pnts, scores, recs, bd_pts):
            # Generate polygon from bounding points if available
            if bd is not None:                
                polygon = self._plot_minimum_covering_polygon(bd)

            # Decode recognized text
            text = self._ctc_decode_recognition(rec)
            if self.voc_size == 37:
                text = text.upper()
            text = "{}".format(text)
            res = {'polygon': [(round(x, 2), round(y, 2)) for x, y in polygon.exterior.coords], 'text': text}
            results.append(res)

        return results  
    

    def _ctc_decode_recognition(self, rec):
        """
        Decode the recognized text from CTC output.

        Args:
            rec (torch.Tensor): The CTC output tensor.

        Returns:
            str: Decoded text string.
        """
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
    
    def _plot_minimum_covering_polygon(self, bboxes):
        """
        Plot the minimum covering polygon for a list of bounding boxes.

        Args:
            bboxes (list): List of bounding boxes, where each bbox is [min_x, min_y, max_x, max_y].

        Returns:
            Polygon: Minimum covering polygon.
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
