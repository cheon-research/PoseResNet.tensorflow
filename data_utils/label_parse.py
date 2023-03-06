"""
2022. 04. 15.
csj@autosemantics.co.kr
"""

import numpy as np

import json
import argparse


path = '/home/cheon/workspace/data/'
label_file = 'sample_test_labels_coco.json'
#plane_labels_file = 'plane_labels.json'
points_labels_file = 'sample_test_labels.json'

with open(path+label_file, 'r') as f:
    data = json.load(f)

#print(data.keys()) # ['info', 'licenses', 'images', 'annotations', 'categories']
#print(data['categories']) # {'id':1, 'name':'plane'}, {'id':2, 'name':'points'}
#print(data['images']) # {'id': 42, 'width': 512, 'height': 512, 'file_name': '11.jpg', 'license': 0, 'flickr_url': '', 'coco_url': '', 'date_captured': 0}
#print(data['annotations']) # {'id': 84, 'image_id': 42, 'category_id': 2, 'segmentation': [[176.74, 334.61, 339.48, 330.95, 257.2, 251.1, 171.25, 326.68]], 'area': 7206.0, 'bbox': [171.25, 251.1, 168.23, 83.51], 'iscrowd': 0, 'attributes': {'occluded': False}}

plane_labels, points_labels = dict(), dict()
#plane_labels['label_type'] = 'plane'
points_labels['label_type'] = 'points'

img_infos = data['images']
img_fnames, img_width = dict(), dict()
for img_info in img_infos:
    id = img_info['id']
    width = img_info['width']
    file_name = img_info['file_name']
    img_fnames[id] = file_name
    img_width[id] = width


annotations = data['annotations']
for ant in annotations:
    img_id = ant['image_id']
    category_id = ant['category_id']
    seg = ant['segmentation'][0] # [min_x1, min_y1, x2, y2, x3, y3, x4, y4]

    points = list()
    for i in range(4):
        p = [ seg[2*i], seg[2*i+1] ]
        points.append(p)
    points = np.array(points, dtype=np.int32)

    if category_id == 1:
        labels = dict()
        sm = points.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
        diff = np.diff(points, axis=1)  # 4쌍의 좌표 각각 x-y 계산
        TL = points[np.argmin(sm)].tolist()  # x+y가 가장 값이 좌상단 좌표, Top Left
        BL = points[np.argmax(diff)].tolist()  # x-y가 가장 큰 값이 좌하단 좌표, Bottom Left
        BR = points[np.argmax(sm)].tolist()  # x+y가 가장 큰 값이 우하단 좌표, Bottom Right
        TR = points[np.argmin(diff)].tolist()  # x-y가 가장 작은 것이 우상단 좌표, Top Right
        points = [ TL, BL, BR, TR ]
        #plane_labels[img_fnames[img_id]] = points
    elif category_id == 2:
        points = np.round((points / img_width[img_id]), 4).tolist() # normalize (position ratio)
        points_labels[img_fnames[img_id]] = points

#if len(plane_labels) > 1:
#    with open(path + plane_labels_file, 'w') as f:
#        json.dump(plane_labels, f)

if len(points_labels_file) > 1:
    with open(path + points_labels_file, 'w') as f:
        json.dump(points_labels, f)
