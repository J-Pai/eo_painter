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
import math

from PIL import Image
from pynput import mouse
from matplotlib import pyplot as plt
from matplotlib import cm


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5
np.set_printoptions(threshold=sys.maxsize)
OFFSET = 15
RANDOM_RANGE = 50

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

def rgb(r, g, b):
    return (r << 16) + (g << 8) + (b)

class ClusterPainter:
    def __init__(self, plot_enable, sim):
        screen_width, screen_height = pyautogui.size()
        self.x_min = 0
        self.x_max = screen_width
        self.y_min = 0
        self.y_max = screen_height
        self.top_left_set = False
        self.bottom_right_set = False
        self.submit_button = None
        self.last_coord = []
        self.plot_enable = plot_enable or sim is not None
        self.sim = sim

    def on_click(self, x, y, button, pressed):
        if self.top_left_set and self.bottom_right_set and not pressed:
            if button == mouse.Button.left:
                self.last_coord = [x, y]
                return False
            else:
                self.last_coord = []
                return False
        if not self.top_left_set and not pressed:
            print('Top Left: {} {}'.format(x, y))
            self.top_left_set = True
            self.x_min = x
            self.y_min = y
            return False
        elif not pressed:
            print('Bottom Right: {} {}'.format(x, y))
            self.bottom_right_set = True
            self.x_max = x
            self.y_max = y
            return False

    def set_bounderies(self):
        if self.sim is not None and len(self.sim) != 0:
            return
        print('SET BOUNDARY : top_left')
        with mouse.Listener(on_click=self.on_click) as listener:
            listener.join()
        print('SET BOUNDARY : bottom_right')
        with mouse.Listener(on_click=self.on_click) as listener:
            listener.join()

        if self.x_max < self.x_min or self.x_max < self.x_min:
            raise ValueError(
                'bottom_right ({}, {}) is less than top_left ({}, {})'.format(
                    self.x_min, self.y_min, self.x_max, self.y_max,))

    def set_submit_button(self):
        if self.sim is not None and len(self.sim) != 0:
            return
        print('SET CONTINUE BUTTON')
        submit_button = pyautogui.locateOnScreen(
            'ui_targets/submit_button.png', confidence=0.5)
        if not submit_button:
            raise ValueError('[Submit] Button not found.')
        self.submit_button = pyautogui.center(submit_button)
        print(self.submit_button)

    def screenshot(self):
        image = pyautogui.screenshot(region=(
            self.x_min,
            self.y_min,
            self.x_max - self.x_min,
            self.y_max - self.y_min,
        ))
        if self.plot_enable and self.sim is None:
            image_name = 'cluster_{}.png'.format(datetime.datetime.now().timestamp())
            image_path = 'images/{}'.format(image_name)
            image.save(image_path)
        return image

    def transform_cluster_coord(self, coord):
        x, y = coord
        return self.x_min + x, self.y_min + y

    def generate_cluster_data(self):
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
            pct = int((r - 255) / -255 * 100)
            return pct

        mean = None
        median = None
        max_pct = None
        min_pct = None

        def cleanup_heat_map(pct):
            if pct < mean:
                return 0
            return pct

        converted_image = None
        if self.sim is not None and len(self.sim) != 0:
            with Image.open(self.sim[0]) as image:
                converted_image = np.array(image.convert('RGB'))
        elif self.sim is not None:
            converted_image = np.array(self.screenshot().convert('RGB'))
        else:
            converted_image = np.array(self.screenshot().convert('RGB'))

        cv_image_array = np.asarray(converted_image, dtype='uint32')
        cv_image_flatten = (cv_image_array[:, :, 0] << 16) \
            + (cv_image_array[:, :, 1] << 8) \
            + (cv_image_array[:, :, 2])
        cv_image_flatten = np.vectorize(ignore_color)(cv_image_flatten)
        cv_image_percentage = np.vectorize(heat_map_percentage)(cv_image_flatten)

        if self.plot_enable:
            plt.subplot(223)
            flatten_plt = plt.imshow(cv_image_flatten)
            flatten_plt.format_cursor_data = lambda data : \
                '[{}]'.format(hex(data))

        cv_image_data = np.array(cv_image_percentage.flatten(), dtype = float)
        cv_image_data[cv_image_data == 0] = np.nan
        mean = np.nanmean(cv_image_data)
        median = np.nanmedian(cv_image_data)
        max_pct = np.nanmax(cv_image_data)
        min_pct = np.nanmin(cv_image_data)
        print(mean, median, max_pct, min_pct)

        cv_image_percentage = np.vectorize(cleanup_heat_map)(cv_image_percentage)

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

        if self.plot_enable:
            plt.subplot(222)
            percentage_plt = plt.imshow(cv_image_percentage)

            plt.subplot(224)
            plt.imshow(converted_image)

        return z, cluster_weight, len(cv_image_percentage[0]), len(cv_image_percentage)

    def execute_clustering(self):
        z, cluster_weight, max_x, max_y = self.generate_cluster_data()

        import hdbscan
        model = hdbscan.HDBSCAN(min_cluster_size = 30, allow_single_cluster = True)
        clusters = model.fit_predict(z) # sample_weight = cluster_weight)
        unique_clusters, counts = np.unique(clusters, return_counts = True)

        if self.plot_enable:
            cluster_plot = plt.subplot(221)
            cluster_plot.axes.invert_yaxis()

        centers_associations = {}
        centers = []
        cluster_counts = dict(zip(unique_clusters, counts))

        sorted_unique_clusters = sorted(unique_clusters, key = lambda x : -cluster_counts[x])
        print(sorted_unique_clusters)

        count = 0
        for cluster in sorted_unique_clusters:
            row_index = np.where(clusters == cluster)
            if count >= 2:
                break
            if len(clusters) > 2 and cluster == -1:
                plt.scatter(z[row_index, 0], z[row_index, 1])
                continue
            x = np.mean(z[row_index, 0])
            y = np.mean(z[row_index, 1])
            centers.append((x, y))
            centers_associations[(x, y)] = z[row_index]
            if self.plot_enable:
                plt.scatter(z[row_index, 0], z[row_index, 1])
            count += 1

        return centers, max_x, max_y

    def generate_edge_to_edge_line(self, slope, b, max_x, max_y):
        y_edge = slope * max_x + b;
        y_zero = b

        x_edge = (max_y - b) / slope
        x_zero = -b / slope

        points = []
        if x_edge >= 0 and x_edge <= max_x:
            points.append([x_edge, max_y])
        if x_zero >= 0 and x_zero <= max_x:
            points.append([x_zero, 0])
        if y_edge >= 0 and y_edge <= max_y:
            points.append([max_x, y_edge])
        if y_zero >= 0 and y_zero <= max_y:
            points.append([0, y_zero])
        return points

    def compute_edge_points(self, main_centers, max_x, max_y):
        main_divider = None
        slope = None
        b = None
        for c in main_centers:
            for other_c in main_centers:
                if c != other_c:
                    x_1, y_1 = c
                    x_2, y_2 = other_c

                    x_mid = (x_1 + x_2) / 2
                    y_mid = (y_1 + y_2) / 2

                    s = (y_2 - y_1) / (x_2 - x_1)

                    slope = -1 / s
                    b = y_mid - slope * x_mid

                    main_divider = np.array(self.generate_edge_to_edge_line(
                        slope, b, max_x, max_y))

                    if self.plot_enable:
                        plt.plot(main_divider[:, 0], main_divider[:, 1], '--', linewidth = 2)

        return main_divider, slope, b

    def action(self):
        while True:
            centers, max_x, max_y = self.execute_clustering()
            divider, slope, b = self.compute_edge_points(
                centers, max_x, max_y)

            print(divider, slope, b)

            if self.plot_enable:
                plt.show()

            if self.sim is not None and len(self.sim) != 0:
                break

            shapes = []
            if divider is None:
                shapes.append([random.randint(0, RANDOM_RANGE), random.randint(0, RANDOM_RANGE)])
                shapes.append([max_x - random.randint(0, RANDOM_RANGE), random.randint(0, RANDOM_RANGE)])
                shapes.append([max_x - random.randint(0, RANDOM_RANGE), max_y - random.randint(0, RANDOM_RANGE)])
                shapes.append([random.randint(0, RANDOM_RANGE), max_y - random.randint(0, RANDOM_RANGE)])
                shapes.append(shapes[0])
            else:
                x1, y1 = divider[0]
                x2, y2 = divider[1]
                top_left = (random.randint(0, RANDOM_RANGE), random.randint(0, RANDOM_RANGE))
                top_right = (max_x - random.randint(0, RANDOM_RANGE), random.randint(0, RANDOM_RANGE))
                bottom_right = (max_x - random.randint(0, RANDOM_RANGE), max_y - random.randint(0, RANDOM_RANGE))
                bottom_left = (random.randint(0, RANDOM_RANGE), max_y - random.randint(0, RANDOM_RANGE))

                offset = math.abs((15 * slope) / 2)

                line1 = self.generate_edge_to_edge_line(slope, b + offset, max_x, max_y)
                line2 = self.generate_edge_to_edge_line(slope, b - offset, max_x, max_y)

                points = [top_left, top_right, bottom_right, bottom_left]
                points.extend(line1)
                points.extend(line2)

                bucket_1 = []
                bucket_2 = []

                for x, y in points:
                    d = (x - x1) * (y2 - y1) - (y - y1) * (x2 -x1)
                    if d < 0:
                        bucket_1.append((x, y))
                    else:
                        bucket_2.append((x, y))

                cx = max_x / 2
                cy = max_y / 2

                def compare(x):
                    return math.atan2(x[0] - cx, x[1] - cy)

                bucket_1 = sorted(bucket_1, key = compare)
                bucket_2 = sorted(bucket_2, key = compare)
                shapes.extend(bucket_1)
                shapes.append(bucket_1[0])
                shapes.extend(bucket_2)
                shapes.append(bucket_2[0])

            count = 0
            for coordinates in shapes:
                x, y = self.transform_cluster_coord(coordinates)
                pyautogui.moveTo(x, y)
                print('RECT 1 Moved mouse to: ({}, {})'.format(x, y))
                pyautogui.click()

            if self.sim is not None and len(self.sim) == 0:
                input('Press [ENTER] when ready.')
                continue

            # Submit entry and click through UI to get next puzzle.
            pyautogui.moveTo(self.submit_button[0], self.submit_button[1])
            pyautogui.click()
            time.sleep(1)
            pyautogui.moveTo(self.submit_button[0], self.submit_button[1])
            pyautogui.click()
            time.sleep(1)
            pyautogui.moveTo(self.submit_button[0], self.submit_button[1])
            pyautogui.click()

            # Wait until UI is ready before starting next iteration.
            continue_button = None
            while not continue_button:
                continue_button = pyautogui.locateOnScreen(
                    'ui_targets/submit_button.png', confidence=0.5)
                time.sleep(1)

        return 'Done!'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', action = 'store_true',
        help = "generate analysis plot prior to doing automated action")
    parser.add_argument('-s', '--simulate', type = str, nargs = '*',
            help = "simulate analysis of passed in image")
    args = parser.parse_args()

    try:
        painter = ClusterPainter(args.plot, args.simulate)
        painter.set_bounderies()
        painter.set_submit_button()

        input('Press [ENTER] when ready.')

        painter.action()
    except pyautogui.FailSafeException:
        return 'PyAutoGUI FailSafe Interrupt'

if __name__ == '__main__':
    print('=== CLUSTER PAINTER ===')
    print('==> Move mouse to top left to stop!')
    print('Exit Message: ', main())
