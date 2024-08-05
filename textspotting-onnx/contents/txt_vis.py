import numpy as np
import cv2
import random

class TextVisualizer:
    def __init__(self, image):
        self.voc_size = 37
        self.CTLABELS = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5',
            '6', '7', '8', '9'
        ]
        self.image = image

    def draw_instance_predictions(self, predictions):
        bd_pts = np.asarray(predictions.bd)
        recs = predictions.recs
        decoded_texts = [self._ctc_decode_recognition(rec) for rec in recs]

        self.overlay_instances(bd_pts, decoded_texts)
        return self.image

    def _ctc_decode_recognition(self, rec):
        last_char = '###'
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    s += self.CTLABELS[c]
                    last_char = c
            else:
                last_char = '###'
        return s

    def draw_polygon(self, segment, color, alpha=0.5):
        """
        Draw a polygon on the image.

        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
        """
        overlay = self.image.copy()
        segment = segment.astype(np.int32)
        cv2.fillPoly(overlay, [segment], color)
        self.image = cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0)

    def overlay_instances(self, bd_pnts, texts, alpha=0.05):
        colors = [
            (0, 128, 0), (0, 192, 0), (255, 0, 255), (192, 0, 192), (128, 0, 128),
            (255, 0, 0), (192, 0, 0), (128, 0, 0), (0, 0, 255), (0, 0, 192),
            (192, 64, 64), (192, 128, 128), (0, 192, 192), (0, 128, 128), (0, 77, 192)
        ]

        for bd_set, text in zip(bd_pnts, texts):
            color = random.choice(colors)            
            for i in range(len(text)):
                if bd_set is not None:
                    #single_bd = bd_set[i]
                    #single_bd = np.reshape(single_bd, (-1, 2))
                    bd = np.hsplit(bd_set, 2)
                    single_bd = np.vstack([bd[0], bd[1][::-1]])                
                    self.draw_polygon(single_bd, color, alpha=alpha)

            text_pos = (int(single_bd[0][0]), int(single_bd[0][1]) - 10)
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

            box_coords = ((text_pos[0], text_pos[1]), (text_pos[0] + text_size[0], text_pos[1] - text_size[1] - 5))
            cv2.rectangle(self.image, box_coords[0], box_coords[1], color, cv2.FILLED)
            cv2.putText(self.image, text, (text_pos[0], text_pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                        (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

