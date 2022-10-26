import os
from typing import Dict, List, Tuple
from sklearn import svm
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
if False:
    cpu_num = 4 # 这里设置成你想运行的CPU个数
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

import sys
sys.path.append('/home/jiahua/yolov5')
from pathlib import Path
from time import sleep, time
from PIL import Image
from PIL.Image import Image as PILImage

import tempfile

from flask import Flask, json, jsonify, request
from flask_cors import CORS

from det_model import yolo_openvino
from feat_extractor import mobnet_openvino
from clf_model import nearCenter, nearNeighbor, svm

import requests
from io import BytesIO
from typing import List, Dict, Tuple



FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
app = Flask('dish_detect')
CORS(app)

global DET_MODEL, FEAT_EXTRACTOR, CLF_MODEL, STORE_IMAGE

@app.route('/detect',methods=['POST'])
def flask_detect():
    tmp_image_path = './tmp.png'
    request.files['img'].save(tmp_image_path)
    imgs = [Image.open(tmp_image_path).convert('RGB')]

    t0 = time()
    bboxes = DET_MODEL.predict(source=tmp_image_path)
    feats = FEAT_EXTRACTOR.det_encode(imgs, bboxes)
    names = CLF_MODEL.det_predict(feats)
    print(f'End of prediction, time: {(time()-t0):.3f}')

    if STORE_IMAGE:
        img_list, label_list = make_img_label_list(imgs, bboxes, names)
        store_images(img_list, label_list)

    # post process
    results = []
    # for each image
    for img_bbox, img_names in zip(bboxes, names):
        box_info = []
        # for each dish
        for bbox, name in zip(img_bbox, img_names):
            box_info.append({
                'class_name': name,  #CLS_MAP[names[row[-1]]],
                'coord': bbox
            })
        # results.append(box_info)
        results.append({'explanations': {'box_info':box_info}})
    return jsonify(results)

@app.route('/detect/bbox',methods=['POST'])
def flask_detect_bbox():
    tmp_image_path = './tmp.png'
    request.files['img'].save(tmp_image_path)

    t0 = time()
    bboxes = DET_MODEL.predict(source=tmp_image_path)
    print(f'end of prediction, time: {(time()-t0):.3f}')

    return jsonify(bboxes)


@app.route('/detect/class',methods=['POST'])
def flask_det_classify():
    # only use step2 and step3 for classify
    tmp_image_path = './tmp.png'
    request.files['img'].save(tmp_image_path)
    imgs = [Image.open(tmp_image_path).convert('RGB')]

    feats = FEAT_EXTRACTOR.encode(imgs)
    names = CLF_MODEL.det_predict([feats])

    return jsonify(names)


@app.route('/clearcache',methods=['GET'])
def clear_clf_model_cache():
    CLF_MODEL.clear_cache()
    return 'clear finished'


@app.route('/train/class',methods=['POST'])
def train_classify():
    # 上传照片，不检测，每张图直接提取特征
    tmp_image_path = './tmp.png'
    request.files['img'].save(tmp_image_path)
    img = [Image.open(tmp_image_path).convert('RGB')]
    label = request.form.get('label')

    feats = FEAT_EXTRACTOR.encode(img)
    CLF_MODEL.train(feats, [label])

    return 'Training Finished'


@app.route('/train/notdet',methods=['POST'])
def train_clf_notdet():
    # 上传本地照片路径，不检测，每张图直接提取特征
    # input: {'food_name':'path_to_img'}
    # img_path -> img -> img_feat -> training clf
    payload = request.json
    train_all = bool(request.args.get('all'))

    imgs = []
    labels = []
    for name, path_list in payload.items():
        for path in path_list:
            imgs.append(Image.open(path).convert('RGB'))
            labels.append(name)

    t0 = time()
    feats = FEAT_EXTRACTOR.encode(imgs)
    CLF_MODEL.train(feats, labels, all=train_all)
    print(f'End of training, time: {(time()-t0):.3f}')

    return 'Training Finished'

@app.route('/train',methods=['POST'])
def train_clf():
    # payload: {'food_name':['path_to_img']}
    # img_path -> img -> img_feat -> training clf
    payload = request.json
    train_all = request.args.get('all')
    paths = []
    labels = []
    imgs = []
    for name, path_list in payload.items():
        for path in path_list:
            imgs.append(Image.open(path).convert('RGB'))
            paths.append(path)
            labels.append(name)

    t0 = time()
    bboxes = DET_MODEL.predict(source=paths)
    feats = FEAT_EXTRACTOR.det_encode(imgs, bboxes)
    # flatten feats and labels to 1-d list
    feats_1d, labels_1d = [], []
    for img_feats, label in zip(feats, labels):
        for feat in img_feats:
            feats_1d.append(feat)
            labels_1d.append(label)

    if train_all:
        CLF_MODEL.train_all(feats_1d, labels_1d)
    else:
        CLF_MODEL.train_increment(feats_1d, labels_1d)

    print(f'End of training, time: {(time()-t0):.3f}')

    # bboxes =
    return jsonify(bboxes)


@app.route('/train/upload',methods=['POST'])
def train_clf_upload():
    # request files['img'] = List[PILImage]
    # request form['label'] = List[str]
    # request form['url'] = List[str]
    train_all = request.args.get('all')
    labels = request.form.getlist('label')
    url_imgs = request.form.getlist('url')
    imgs = request.files.getlist('img')

    res = request.json
    if not res:
        return 'No json format data received'
    else:
        train_all = res['action'] == 'all' # all or add
        labels, url_imgs = unpair_kv(res['pics'])


    # save uploaded image and get paths
    dst_folder = tempfile.TemporaryDirectory().name
    os.makedirs(dst_folder, exist_ok=True)
    img_paths = []

    if len(imgs)>0:
        for i, im in enumerate(imgs):
            dst_path = f'{dst_folder}/{i}.png'
            im.save(dst_path)
            img_paths.append(dst_path)
            imgs = [Image.open(path).convert('RGB') for path in img_paths]

    elif len(url_imgs)>0:
        imgs = load_url_imgs(url_imgs)
        for i, im in enumerate(imgs):
            dst_path = f'{dst_folder}/{i}.png'
            im.save(dst_path)
            img_paths.append(dst_path)

    # imgs -> bboxes -> cropped imgs -> feats + labels -> train CLF_MODEL
    t0 = time()
    bboxes = DET_MODEL.predict(source=img_paths)
    feats = FEAT_EXTRACTOR.det_encode(imgs, bboxes)
    # flatten feats and labels to 1-d list
    feats_1d, labels_1d = [], []
    for img_feats, label in zip(feats, labels):
        for feat in img_feats:
            feats_1d.append(feat)
            labels_1d.append(label)

    if train_all:
        CLF_MODEL.train_all(feats_1d, labels_1d)
    else:
        CLF_MODEL.train_increment(feats_1d, labels_1d)

    if STORE_IMAGE:
        full_labels = []
        for img_bbox, img_label in zip(bboxes, labels):
            full_labels.append([img_label] * len(img_bbox))

        img_list, label_list = make_img_label_list(imgs, bboxes, full_labels)
        store_images(img_list, label_list)

    print(f'End of training, time: {(time()-t0):.3f}')
    return jsonify(bboxes)


@app.route('/label', methods=['GET'])
def get_lable():
    labels = CLF_MODEL.get_label()
    return jsonify(labels)

def make_img_label_list(images, bboxes, labels):
    img_list = []
    label_list = []
    for img, bbox, label in zip(images, bboxes, labels):
        for box, l in zip(bbox, label):
            img_list.append(img.crop(box))
            label_list.append(l)
    return img_list, label_list

def store_images(imgs:List[PILImage], labels:List[Tuple[int, str]]):
    folder_path = './stored_imgs'
    os.makedirs(folder_path, exist_ok=True)

    labels = [str(l) for l in labels]
    for im, l in zip(imgs, labels):
        dst_folder_path = os.path.join(folder_path, l)
        os.makedirs(dst_folder_path, exist_ok=True)

        dst_path = os.path.join(dst_folder_path, str(time()) + '.jpeg')
        im.save(dst_path)

    return

def load_url_imgs(url_list:List[str]):
    img_list = []
    for url in url_list:
        img_bytes = requests.get(url).content
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        img_list.append(img)
    return img_list


def unpair_kv(key_values: Dict[str,List[str]]) -> Tuple[List, List]:
    # convert key-List data to one-to-one correspondence
    all_keys = []
    all_values = []
    for key, values in key_values.items():
        for v in values:
            all_keys.append(key)
            all_values.append(v)
    return all_keys, all_values


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = os.path.join(os.path.dirname(__file__),'weights')

    # DET_MODEL = yolo(
    #     weights=os.path.join(weights_path, 'yolov5_best.pt'),
    #     imgsz=720,
    #     device=device)

    # fixed image size = 736
    DET_MODEL = yolo_openvino(
        xml_path = os.path.join(weights_path, 'yolov5_best.xml'),
        bin_path = os.path.join(weights_path, 'yolov5_best.bin'),
    )

    # FEAT_EXTRACTOR = resnext(
    #     weights='./weights/resnext_food783.pth',
    #     num_classes=783,
    #     imgsz=299,
    #     device=device,
    #     half=True
    # )

    # FEAT_EXTRACTOR = resnext_pretrain(
    #     weights= os.path.join(weights_path, 'distil_mobv3_run1_best.pth'),
    #     base_model=mobilenet_v3_large(pretrained=False),
    #     imgsz=299,
    #     device=device,
    #     half=False # CPU not support fp16
    # )

    FEAT_EXTRACTOR = mobnet_openvino(
        xml_path = os.path.join(weights_path, 'distil_mobv3_run1_best.xml'),
        bin_path = os.path.join(weights_path, 'distil_mobv3_run1_best.bin'),
    )


    CLF_MODEL = nearCenter(
        cache_path= os.path.join(weights_path, 'clf_model_0917_new.pkl')
        # aux_clf_path=os.path.join(weights_path, 'distil_mobv3_clf_layer.pth'),
        # aux_id_map=os.path.join(weights_path, 'id_name_map.json')
    )
    # CLF_MODEL = nearNeighbor(cache_path='/home/jiahua/yolov5/weights/clf_model_0830_spec.pkl')
    # CLF_MODEL = svm(cache_path=os.path.join(weights_path, 'clf_model_0917_new.pkl'))

    # check if feature length of extractor and classifer is the same
    assert (CLF_MODEL.feat_length is None) or (
        FEAT_EXTRACTOR.feat_length == CLF_MODEL.feat_length), (
            "The extractor's feature length is not matching with classifier")

    STORE_IMAGE = True


    #TODO argparth ip and host

    app.run(host='0.0.0.0',
        port=36188,
        debug=True,
        use_reloader=False)
