import argparse
import cv2 as cv2
import datetime
import json
import numpy
import pyautogui
import random
import sys
import time
import os

from pynput import mouse
from matplotlib import pyplot as plt

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5
OFFSET = 10
IMAGE_FILEPATH = 'images/cluster_1618005314.143233.png'

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
        pass

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
