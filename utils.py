import cv2

import json


def loadJSON(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data


def draw_points(img, points, points_labels=None, color=(0, 255, 0), with_lines=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, point in enumerate(points):
        img = cv2.circle(img, point, 2, color, -1)
        if points_labels:
            img = cv2.putText(img, points_labels[idx], point, font, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

    if with_lines:
        for i in len(points):
            if i == (len(points) - 1):
                img = cv2.line(img, points[i], points[0], (0, 255, 0), 2)
            else:
                img = cv2.line(img, points[i], points[i+1], (0, 255, 0), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
