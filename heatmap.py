from collections import deque
import numpy as np
import cv2

class Heatmapper:
    def __init__(self,memory=12):
        self._memory = deque(maxlen=memory)
 
    def update(self, hot_windows):
        self._memory.append(hot_windows)

    def get_heatmap(self, img):
        heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)

        for spots in self._memory:
            for spot in spots:
                heatmap[spot[0][1]:spot[1][1], spot[0][0]:spot[1][0]] += 1
        return heatmap

    def apply_threshold(self,heatmap, threshold=15):
        heatmap[heatmap < threshold] = 0
        return heatmap


