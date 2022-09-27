from flask import Flask, json, jsonify, request, g
from flask_cors import CORS

import os
from video_sampler import VideoSampler, if_background

from data_storage import Box, SubImage, MainImage, xyxy2xywh
from PIL import Image, ImageDraw
import shutil
from threading import Thread
import pickle
from time import time
from math import ceil
from typing import List

app = Flask(__name__)
CORS(app)
# BASE_IP = '192.168.2.62'
BASE_IP = '127.0.0.1'
BASE_PORT = '36192'
ADDRESS = f'http://{BASE_IP}:{BASE_PORT}'
NUM_PER_PAGE = 25

video_folder = '/Users/jiahua/Downloads/moving_det_cv/testdata'
tmp_folder = '/Users/jiahua/Downloads/moving_det_cv/tmp'
STATIC_PATH = './static'
os.makedirs(tmp_folder, exist_ok=True)

# prepare model
from detect_api.clf_model_simple import nearCenter
from detect_api.feat_extractor import mobnet_openvino
from detect_api.det_model import yolo_openvino
import cv2
import numpy as np
det_model = yolo_openvino(
    xml_path = os.path.join('/Users/jiahua/Downloads/moving_det_cv/detect_api/weights/yolov5_best.xml'),
    bin_path = os.path.join('/Users/jiahua/Downloads/moving_det_cv/detect_api/weights/yolov5_best.bin'),
)
extractor = mobnet_openvino(
    bin_path='/Users/jiahua/Downloads/moving_det_cv/detect_api/weights/distil_mobv3_run1_best.bin',
    xml_path='/Users/jiahua/Downloads/moving_det_cv/detect_api/weights/distil_mobv3_run1_best.xml'
)
classifier = nearCenter()

# data storage to manage images and processing result
from data_storage import DataStorage
ds = DataStorage()

@app.route('/get_video_paths', methods=['GET'])
def get_video_paths():
    video_list = [
        os.path.join(video_folder, name) for name in os.listdir(video_folder)
        if name.endswith('.mp4')
    ]
    # json.dumps(video_list)
    return jsonify(video_list)

@app.route('/get_directories', methods=['GET'])
def get_directories():
    dir_list = [
        os.path.join(tmp_folder, name) for name in os.listdir(tmp_folder)
        if os.path.isdir(os.path.join(tmp_folder, name))
    ]
    dir_list = sorted(dir_list)
    return jsonify(dir_list)

@app.route('/process_video', methods=['GET'])
def process_video():
    video_path = request.args.get('path', None)
    fps = int(request.args.get('fps', 15))
    assume_stable = int(request.args.get('stable', 900))
    diff_area = float(request.args.get('diff_area', 0.2))

    if not (video_path and os.path.isfile(video_path)
            and video_path.endswith('mp4')):
        return 'The path is not a video file', 500

    g.video_path = video_path
    g.video_name = os.path.splitext(os.path.basename(video_path))[0]

    if not hasattr(g, 'vs'):
        g.vs = VideoSampler()

    g.vs.capture(src=g.video_path,
                 fps=fps,
                 assume_stable=assume_stable,
                 diff_area_base=diff_area,
                 video_name=g.video_name)

    # save to temp directory
    save_path = os.path.join(tmp_folder, g.video_name)
    os.makedirs(save_path, exist_ok=True)
    clear_dir(save_path)
    g.vs.save_to_folder(root_path=save_path)

    return 'waiting', 200

@app.route('/save_result', methods=['GET'])
def save_capture_result():
    save_path = request.args.get('path', None)
    assert save_path and os.path.isdir(save_path), 'The path is not a folder'
    save_path = os.path.join(save_path, g.video_name)
    os.makedirs(save_path, exist_ok=True)

    g.vs.save_to_folder(root_path=save_path)
    return 'success', 200

def get_symlink_path(folder_path):
    # create symlink in tmp folder
    symlink_path = os.path.join(STATIC_PATH, os.path.basename(folder_path))
    if not os.path.exists(symlink_path):
        os.symlink(folder_path, symlink_path)
    return symlink_path


@app.route('/get_mainimg', methods=['POST'])
def get_mainimg():
    folder_path = request.json.get('path', None)
    # page = request.json.get('page', 1)

    # load this image folder into datastorage
    symlink_path = get_symlink_path(folder_path)
    ds.load_datafolder(symlink_path)

    # prepare the main images to show
    mainimg_data = []
    for mainimg_id, mainimg in ds.mainimgs.items():
        # use preview image if exist
        if os.path.exists(mainimg.preview_path):
            path = path2url(mainimg.preview_path, add_time=True)
        else:
            path = path2url(mainimg.path, add_time=True)
        mainimg_data.append({'key': mainimg_id, 'path': path})
    # sort by the names of image for user to compare between each
    mainimg_data = list(sorted(mainimg_data, key=lambda x: x['key']))

    # count = len(mainimg_data)
    # page_num = ceil(count/NUM_PER_PAGE)
    # page = min(page, page_num)-1

    # mainimg_data = mainimg_data[page * NUM_PER_PAGE:(page + 1) * NUM_PER_PAGE]
    # page_num = 1

    return jsonify({'mainimg': mainimg_data})


@app.route('/get_subimg', methods=['POST'])
def get_subimg():
    folder_path = request.json.get('path', None)
    selected_labels = request.json.get('labels', []) # each element is class names

    # load this image folder into datastorage
    symlink_path = get_symlink_path(folder_path)
    ds.load_datafolder(symlink_path)

    # prepare the sub images to show
    subimg_data = []
    confirmed_data = []
    subimg: SubImage
    for subimg_id, subimg in ds.subimgs.items():
        name = ds.id2name.get(subimg.cls_id, '未分类')
        if len(selected_labels) > 0 and (name not in selected_labels):
            continue
        else:
            path = path2url(subimg.path, add_time=True)
            data = {'key': subimg_id, 'path': path, 'class': name}
            if subimg.confirmed:
                confirmed_data.append(data)
            else:
                subimg_data.append(data)


    return jsonify({
        'subimg': subimg_data,
        'confirmed': confirmed_data,
        # 'page_num': page
    })


@app.route('/delete_images', methods=['POST'])
def delete_images():
    image_keys = request.json.get('path', None)
    for key in image_keys:
        ds.delete_mainimg(key)
    return 'success', 200

@app.route('/find_similar', methods=['POST'])
def find_similar_img():
    image_keys = request.json.get('keys', None)
    fps = request.json.get('fps', 15)
    diff_area_base = request.json.get('diff_area', 0.1)
    move_image_keys = []
    for key in ds.mainimgs.keys():
        img_path = ds.mainimgs[key].path
        for bg_img_key in image_keys:
            if bg_img_key == key:
                continue
            bg_img_path = ds.mainimgs[bg_img_key].path
            move = if_background(img_path, bg_img_path, fps=fps, diff_area_base=diff_area_base)

            if move == 0: # have found similar image
                move_image_keys.append(key)
                break

    similar_data = []
    for mainimg_id in move_image_keys + image_keys:
        path = path2url(ds.mainimgs[mainimg_id].path, add_time=True)
        similar_data.append({'key': mainimg_id, 'path': path})

    return jsonify(similar_data)


@app.route('/detect_box', methods=['POST'])
def detect_box():
    mainimg_ids = request.json.get('path', None)
    image_paths = [ds.mainimgs[id].path for id in mainimg_ids]
    bboxes = det_model.predict(source=image_paths)

    for mainimg_id, bbox in zip(mainimg_ids, bboxes):
        mainimg: MainImage = ds.mainimgs[mainimg_id]

        boxes = [] # make box objects
        for x1, y1, x2, y2 in bbox:
            width, height = Image.open(mainimg.path).size
            boxes.append(Box(x1, y1, x2, y2, width, height))

        # refresh subimgs of main image object
        ds.reset_mainimg(mainimg_id)
        new_subimgs = mainimg.add_boxes(boxes)

        # refresh subimgs of data storage
        ds.subimgs.update(new_subimgs)

    return 'waiting', 200


@app.route('/add_cls_name', methods=['GET'])
def add_cls_name():
    cls_name = request.args.get('cls_name', None)
    if cls_name is None:
        return 'no `cls_name` argument', 500
    ds.add_cls(new_cls_name=cls_name)
    return 'ok', 200


@app.route('/get_cls_list', methods=['GET'])
def get_cls_list():
    return jsonify(list(ds.name2id.keys()))

@app.route('/set_cls_name', methods=['POST'])
def set_cls_name():
    subimg_keys = request.json.get('key', None)
    cls_name = request.json.get('cls_name', None)
    ds.add_cls(new_cls_name=cls_name)
    for key in subimg_keys:
        ds.set_subimg_cls(key, int(ds.name2id[cls_name]))
    return 'ok', 200

@app.route('/confirm_subimgs', methods=['POST'])
def confirm_subimgs():
    subimg_keys = request.json.get('key', None)
    ds.confirm_subimgs(subimg_keys)
    return 'ok', 200

@app.route('/auto_label_subimg', methods=['GET'])
def auto_label_subimg():
    # use all confirmed subimg as training set
    feats = []
    labels = []
    not_confirmed_subimg_feats = {}

    for subimg_id, subimg in ds.subimgs.items():
        feat = subimg.get_feat(extractor)
        if subimg.confirmed:
            feats.append(feat)
            labels.append(subimg.cls_id)
        else:
            not_confirmed_subimg_feats[subimg_id] = feat

    if len(set(labels)) < 2:
        return '请多标注几个类别', 200

    classifier.train_all(data = feats, label=labels)

    # predict not confirmed images
    for subimg_id, feat in not_confirmed_subimg_feats.items():
        new_label = classifier.predict([feat])[0]
        # modify subimgs and corresponding files
        ds.set_subimg_cls(subimg_id, new_cls_id=new_label, confirm=False)

    return '自动标注完成，请确认', 200

@app.route('/delet_subimgs', methods=['POST'])
def delete_subimgs():
    keys = request.json.get('key', None)
    ds.delete_subimgs(keys)

    return 'success', 200

@app.route('/analysis/counting_class', methods=['GET'])
def counting_class():
    # count the number of sub images of each class
    counter = {}
    subimg: SubImage
    for subimg_id, subimg in ds.subimgs.items():
        if subimg.cls_id in counter:
            counter[subimg.cls_id] +=1
        else:
            counter[subimg.cls_id] = 1

    # convert class id to class name
    counter = {ds.id2name[k]: v for k, v in counter.items()}
    return counter

def path2url(path, add_time=False):
    # For some resource need inplace changing, add timestamp to
    # force the browser to refresh them
    url = path.replace(STATIC_PATH, f'{ADDRESS}/static')
    if add_time:
        url = url + f'?t={time()}'
    return url

def get_image_key(image_path):
    return os.path.basename(image_path).replace('_preview', '').replace('.jpg', '')


def det_res2bbox(det_res, image_paths):
    for det, path in zip(det_res, image_paths):
        img = Image.open(path)
        img_name = os.path.splitext(os.path.basename(path))[0]
        dir_path = os.path.join(os.path.dirname(path), 'sub_images')
        os.makedirs(dir_path, exist_ok=True)
        width, height = img.size
        boxes = []
        subimg_keys = []
        for i, (x1, y1, x2, y2) in enumerate(det):
            x, y, w, h = xyxy2xywh(x1, y1, x2, y2, width, height)
            boxes.append(Box(x, y, w, h, cls=-1))

            # crop the subimage
            si = img.crop([x1, y1, x2, y2])
            si_path = os.path.join(dir_path, f'{img_name}_{i}.jpg')
            si.save(si_path)

            # save this subimage to data storage
            subimg = SubImage(si_path,
                         cls_id=-1,
                         parent_image=ds.mainimgs[img_name].path)
            ds.subimgs[f'{img_name}_{i}'] = subimg
            subimg_keys.append(f'{img_name}_{i}')

        ds.mainimgs[img_name].boxes = boxes
        ds.mainimgs[img_name].subimages = subimg_keys

        with open(path.replace('.jpg', '.txt'), 'w') as f:
            f.writelines([box.to_str() + '\n' for box in boxes if box])

        # draw the bounding box on main image as preview
        dir_path = os.path.join(os.path.dirname(path), 'preview')
        os.makedirs(dir_path, exist_ok=True)
        draw = ImageDraw.Draw(img)
        for i, (x1, y1, x2, y2) in enumerate(det):
            draw.rectangle((x1,y1,x2,y2), outline='red', width=3)
        preview_path =  os.path.join(dir_path, f'{img_name}_preview.jpg')
        img.save(preview_path)


def clear_dir(path):
    shutil.rmtree(path)
    os.makedirs(path)
    # for name in os.listdir(path):
    #     if name.endswith('.jpg'):
    #         os.remove(os.path.join(path, name))


if __name__ == '__main__':

    app.run(host=BASE_IP, port=BASE_PORT, debug=True, use_reloader=False)
