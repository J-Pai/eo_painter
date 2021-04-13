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
    0x080808,
    0x292929,
    0x020202,
    0x0c0c0c,
    0x222222,
    0x1c1c1c,
    0x040404,
    0x202120,
    0x100,
    0x0f0f0f,
    0x272827,
    0x10001,
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

    def screenshot(self):
        image = pyautogui.screenshot(region=(
            self.x_min,
            self.y_min,
            self.x_max - self.x_min,
            self.y_max - self.y_min,
        ))
        # image_name = "cluster_{}.png".format(datetime.datetime.now().timestamp())
        # image_path = "images/{}".format(image_name)
        # image.save(image_path)
        return image

    def transform_cluster_coord(self, coord):
        x, y = coord
        return self.x_min + x, self.y_min + y

    def action(self):
        # Ignore square dots in the corners.
        def ignore_color(c):
            if c in BLACK_LISTED_COLORS:
                return 0
            return c

        def heat_map_percentage(c):
            r = (c >> 16) & 0xFF;
            g = (c >> 8) & 0xFF;
            b = c & 0xFF;
            if r == 0 and g == 0 and b == 0:
                return 0
            value = (r / 255 + 2) + (g / 255 + 1) + b / 255
            return np.uint32(value / 3 * 100)

        # with Image.open(self.screenshot()) as image:
        # with Image.open("images\cluster_1618287905.126905.png") as image:
            # converted_image = np.array(image.convert('RGB'))
        converted_image = np.array(self.screenshot().convert('RGB'))

        cv_image_array = np.asarray(converted_image, dtype='uint32')
        cv_image_flatten = (cv_image_array[:, :, 0] << 16) \
            + (cv_image_array[:, :, 1] << 8) \
            + (cv_image_array[:, :, 0])
        cv_image_flatten = np.vectorize(ignore_color)(cv_image_flatten)

        cv_image_percentage = np.vectorize(heat_map_percentage)(cv_image_flatten)

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
        model = KMeans(n_clusters = 2)
        clusters = model.fit_predict(z, sample_weight = cluster_weight)
        unique_clusters = np.unique(clusters)
        centers = model.cluster_centers_

        cluster_plot = plt.subplot(221)
        centers_associations = {}
        for cluster in unique_clusters:
            row_index = np.where(clusters == cluster)
            # scatter = plt.scatter(z[row_index, 0], z[row_index, 1])
            x, y = centers[cluster]
            centers_associations[(x, y)] = z[row_index]

        from sklearn.cluster import MeanShift
        center_model = MeanShift()
        center_clusters = center_model.fit_predict(centers)
        unique_center_clusters = np.unique(center_clusters)

        main_centers = []
        scatter = None
        for cluster in unique_center_clusters:
            row_index = np.where(center_clusters == cluster)
            points = []
            for x, y in centers[row_index]:
                if len(points) == 0:
                    points = centers_associations[(x, y)]
                else:
                    points = np.append(points, centers_associations[(x, y)], axis = 0)
            scatter = plt.scatter(points[:, 0], points[:, 1])
            plt.scatter(centers[row_index, 0], centers[row_index, 1],
                        s = 80, marker = 's')
            main_centers.append([np.mean(points[:, 0]), np.mean(points[:, 1])])
            plt.scatter(main_centers[cluster][0], main_centers[cluster][1], marker = 'D')

        bisectors = None
        max_x = len(cv_image_percentage[0])
        max_y = len(cv_image_percentage)
        bisectors = []
        top_edge = -1
        right_edge = -1
        bottom_edge = -1
        left_edge = -1
        for c in main_centers:
            for other_c in main_centers:
                if c != other_c:
                    x_1, y_1 = c
                    x_2, y_2 = other_c

                    x_mid = (x_1 + x_2) / 2
                    y_mid = (y_1 + y_2) / 2

                    slope = (y_2 - y_1) / (x_2 - x_1)

                    perp_slope = -1 / slope
                    b = y_mid - perp_slope * x_mid

                    y_edge = perp_slope * max_x + b;
                    y_zero = b

                    x_edge = (max_y - b) / perp_slope
                    x_zero = -b / perp_slope

                    points = []
                    if x_edge >= 0 and x_edge <= max_x:
                        bottom_edge = x_edge
                        points.append([x_edge, max_y])
                    if x_zero >= 0 and x_zero <= max_x:
                        top_edge = x_zero
                        points.append([x_zero, 0])
                    if y_edge >= 0 and y_edge <= max_y:
                        right_edge = y_edge
                        points.append([max_x, y_edge])
                    if y_zero >= 0 and y_zero <= max_y:
                        left_edge = y_zero
                        points.append([0, y_zero])

                    np_points = np.array(points)
                    plt.plot(np_points[:, 0], np_points[:, 1], '--', linewidth = 2)
                    bisectors.append(np_points)

        bisectors = np.array(bisectors)

        cluster_plot.axes.invert_yaxis()
        plt.subplot(222)
        percentage_plt = plt.imshow(cv_image_percentage)

        plt.subplot(223)
        flatten_plt = plt.imshow(cv_image_flatten)
        flatten_plt.format_cursor_data = lambda data : \
             "[{}]".format(hex(data))

        plt.subplot(224)
        plt.imshow(converted_image)

        plt.show()

        print(top_edge, right_edge, bottom_edge, left_edge)

        shapes = []
        shapes.append([0, 0])
        if top_edge != -1:
            shapes.append([top_edge, 0])
            shapes.append([bottom_edge, max_y])
            shapes.append([0, max_y])
            shapes.append([0, 0])

            shapes.append([top_edge + 15, 0])
            shapes.append([max_x, 0])
            shapes.append([max_x, max_y])
            shapes.append([bottom_edge + 15, max_y])
            shapes.append([top_edge + 15, 0])
        else:
            shapes.append([max_x, 0])
            shapes.append([max_x, right_edge])
            shapes.append([0 ,left_edge])
            shapes.append([0, 0])

            shapes.append([0, left_edge + 15])
            shapes.append([max_x, right_edge + 15])
            shapes.append([max_x, max_y])
            shapes.append([0, max_y])
            shapes.append([0, left_edge + 15])

        print(shapes)

        for coordinates in shapes:
            x, y = self.transform_cluster_coord(coordinates)
            pyautogui.moveTo(x, y)
            print("RECT 1 Moved mouse to: ({}, {})".format(x, y))
            pyautogui.click()

        return "Done!"

def main():
    try:
        painter = ClusterPainter()
        painter.set_bounderies()
        painter.set_submit_button()

        input("Press [ENTER] when ready.")

        painter.action()
    except pyautogui.FailSafeException:
        return "PyAutoGUI FailSafe Interrupt"

if __name__ == "__main__":
    print("=== CLUSTER PAINTER ===")
    print("==> Move mouse to top left to stop!")
    print("Exit Message: ", main())
