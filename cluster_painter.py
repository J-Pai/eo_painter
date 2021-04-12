import argparse
import cv2 as cv
import datetime
import json
import numpy as np
import pyautogui
import random
import sys
import time
import os

from PIL import Image
from pynput import mouse
from matplotlib import pyplot as plt
from matplotlib import cm


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5
np.set_printoptions(threshold=sys.maxsize)
OFFSET = 10
IMAGE_FILEPATH = "images/cluster_1618005814.455007.png"
BLACK_LISTED_COLORS = {
    0x010101,
    0x282828,
    0x212121,
    0x1b1b1b,
    0x0b0b0b,
    0x272727,
    0x202020,
    0x1a1a1a,
    0x0a0a0a,
    0x090909,
    0x030303,
}

class ClusterPainter:
    def __init__(self):
        screen_width, screen_height = pyautogui.size()
        print("Screen Information: ", screen_width, screen_height)
        self.x_min = 0
        self.x_max = screen_width
        self.y_min = 0
        self.y_max = screen_height
        self.top_left_set = False
        self.bottom_right_set = False
        self.submit_button = None
        self.last_coord = []


    def on_click(self, x, y, button, pressed):
        if self.top_left_set and self.bottom_right_set and not pressed:
            if button == mouse.Button.left:
                self.last_coord = [x, y]
                return False
            else:
                self.last_coord = []
                return False
        if not self.top_left_set and not pressed:
            print("Top Left: {} {}".format(x, y))
            self.top_left_set = True
            self.x_min = x
            self.y_min = y
            return False
        elif not pressed:
            print("Bottom Right: {} {}".format(x, y))
            self.bottom_right_set = True
            self.x_max = x
            self.y_max = y
            return False

    def set_bounderies(self):
        print("SET BOUNDARY : top_left")
        with mouse.Listener(on_click=self.on_click) as listener:
            listener.join()
        print("SET BOUNDARY : bottom_right")
        with mouse.Listener(on_click=self.on_click) as listener:
            listener.join()

        if self.x_max < self.x_min or self.x_max < self.x_min:
            raise ValueError(
                "bottom_right ({}, {}) is less than top_left ({}, {})".format(
                    self.x_min, self.y_min, self.x_max, self.y_max,))

    def set_submit_button(self):
        print("SET CONTINUE BUTTON")
        submit_button = pyautogui.locateOnScreen(
            'ui_targets/submit_button.png', confidence=0.5)
        if not submit_button:
            raise ValueError("[Submit] Button not found.")
        self.submit_button = pyautogui.center(submit_button)
        print(self.submit_button)

    def action(self):
        # Ignore square dots in the corners.
        def ignore_color(c):
            if c <= 15:
                return 0
            return c

        with Image.open("images\cluster_1618013627.190346.png") as image:
            converted_image = np.array(image.convert('RGB'))

        cv_image_array = np.asarray(converted_image, dtype='uint32')
        cv_image_flatten = (cv_image_array[:, :, 0] << 16) \
            + (cv_image_array[:, :, 1] << 8) \
            + (cv_image_array[:, :, 0])


        cv_image_percentage = np.array(list(map(lambda c : \
                                                np.uint32(c/0xFFFFFF*100), \
                                                cv_image_flatten)))
        cv_image_percentage = np.vectorize(ignore_color)(cv_image_percentage)

        cluster_x = []
        cluster_y = []
        cluster_weight = []
        for y in range(len(cv_image_percentage)):
            for x in range(len(cv_image_percentage[y])):
                pct = cv_image_percentage[y][x]
                # for j in range(pct):
                if pct != 0:
                    cluster_x.append(x)
                    cluster_y.append(y)
                    cluster_weight.append(pct)

        z = np.vstack((cluster_x, cluster_y)).T
        z = np.float32(z)

        from sklearn.cluster import KMeans
        model = KMeans(n_clusters = 8)
        clusters = model.fit_predict(z, sample_weight = cluster_weight)
        unique_clusters = np.unique(clusters)
        centers = model.cluster_centers_

        cluster_plot = plt.subplot(121)
        for cluster in unique_clusters:
            row_index = np.where(clusters == cluster)
            scatter = plt.scatter(z[row_index, 0], z[row_index, 1])

        from sklearn.cluster import DBSCAN
        center_model = DBSCAN(eps = 0.3 * len(cv_image_percentage))
        center_clusters = center_model.fit_predict(centers)
        unique_center_clusters = np.unique(center_clusters)
        for cluster in unique_center_clusters:
            row_index = np.where(center_clusters == cluster)
            plt.scatter(centers[row_index, 0], centers[row_index, 1],
                        s = 80, marker = 's')


        cluster_plot.axes.invert_yaxis()
        plt.subplot(122)
        percentage_plt = plt.imshow(cv_image_percentage)
        plt.show()

        return "Done!"

def main():
    try:
        painter = ClusterPainter()
        # painter.set_bounderies()
        # painter.set_submit_button()
        input("Press [ENTER] when ready.")
        painter.action()
    except pyautogui.FailSafeException:
        return "PyAutoGUI FailSafe Interrupt"

if __name__ == "__main__":
    print("=== CLUSTER PAINTER ===")
    print("==> Move mouse to top left to stop!")
    print("Exit Message: ", main())
