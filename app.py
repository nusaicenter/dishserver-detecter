from flask import Flask, json, jsonify, request, g
from flask_cors import CORS

import os
from video_sampler import VideoSampler

from data_storage import Box, SubImage, MainImage
from PIL import Image, ImageDraw

app = Flask('data manage')
CORS(app)
BASE_IP = '127.0.0.1'
BASE_PORT = '36192'
address = f'http://{BASE_IP}:{BASE_PORT}'

video_folder = '/Users/jiahua/Downloads/dishvideo_220725'
tmp_folder = '/Users/jiahua/Downloads/moving_det_cv/tmp'
static_path = './static'
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
    save_capture_tmp(tmp_folder)
    return 'waiting', 200

@app.route('/save_result', methods=['GET'])
def save_capture_result():
    save_path = request.args.get('path', None)
    assert save_path and os.path.isdir(save_path), 'The path is not a folder'
    save_path = os.path.join(save_path, g.video_name)
    os.makedirs(save_path, exist_ok=True)

    g.vs.save_to_folder(root_path=save_path)
    return 'success', 200

@app.route('/get_images', methods=['GET'])
def get_images_in_directory():
    folder_path = request.args.get('path', None)
    # create symlink in tmp folder
    symlink_path = os.path.join(static_path, os.path.basename(folder_path))
    if not os.path.exists(symlink_path):
        os.symlink(folder_path, symlink_path)
    img_path_list = [
        os.path.join(symlink_path, name) for name in os.listdir(symlink_path)
        if name.endswith('.jpg')
    ]

    # use preview image if exist
    for i, _ in enumerate(img_path_list):
        origin_path = img_path_list[i]
        preview_path = os.path.join(
            os.path.dirname(origin_path), 'preview',
            os.path.basename(origin_path).replace('.jpg', '_preview.jpg'))
        if os.path.exists(preview_path):
            img_path_list[i] = preview_path

    img_path_list = [path.replace(static_path, f'{address}/static') for path in img_path_list ]

    # load this image folder into datastorage
    ds.load_datafolder(symlink_path)

    # img_keys = [get_image_key(path) for path in img_path_list]

    subimg_path_list = []

    return jsonify([{
        'key': get_image_key(path),
        'path': path
    } for path in img_path_list])


@app.route('/detect_box', methods=['POST'])
def detect_box():
    image_keys = request.json.get('path',None)
    image_paths = [ds.mainimgs[k].path for k in image_keys]
    bboxes = det_model.predict(source=image_paths)
    det_res2bbox(bboxes, image_paths)
    return 'waiting', 200


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
        subimgs = []

        for i, (x1, y1, x2, y2) in enumerate(det):
            x = (x1 + x2) / 2 / width
            y = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
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
            subimgs.append(subimg)

        ds.mainimgs[img_name].boxes = boxes
        ds.mainimgs[img_name].subimages = subimgs

        # TODO save yolo format txt label

        with open(path.replace('.jpg', '.txt'), 'w') as f:
            f.writelines([box.toStr() + '\n' for box in boxes])

        # draw the bounding box on main image as preview
        dir_path = os.path.join(os.path.dirname(path), 'preview')
        os.makedirs(dir_path, exist_ok=True)
        draw = ImageDraw.Draw(img)
        for i, (x1, y1, x2, y2) in enumerate(det):
            draw.rectangle((x1,y1,x2,y2), outline='green')
        preview_path =  os.path.join(dir_path, f'{img_name}_preview.jpg')
        img.save(preview_path)




def save_capture_tmp(path):
    save_path = os.path.join(path, g.video_name)
    os.makedirs(save_path, exist_ok=True)
    g.vs.save_to_folder(root_path=save_path)


def clear_dir(path):
    for name in os.listdir(path):
        os.remove(os.path.join(path, name))


if __name__ == '__main__':

    app.run(host=BASE_IP, port=BASE_PORT, debug=True, use_reloader=False)
