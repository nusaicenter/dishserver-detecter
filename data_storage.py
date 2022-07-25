from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import os
from os.path import join, basename, isdir, isfile, splitext
from PIL.Image import Image as PILImage
import json
import pickle

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class Box():
    def __init__(self, x, y, w, h, cls=-1) -> None:
        # -1 means unknown class
        self.x: float = x
        self.y: float = y
        self.w: float = w
        self.h: float = h
        self.cls: int = cls


class MainImage(object):
    def __init__(self, path, preview_path=None, boxes=None, subimages=None) -> None:
        check_image_file(path)
        self.path: str = path
        self.preview_path = preview_path if preview_path else ''
        # self.project: str = project
        self.boxes: List[Box] = boxes if boxes else list()
        if subimages:
            assert len(subimages) == len(boxes), 'number of subimages and boxes not match'
            self.subimages: List[str] = subimages
        else:
            self.subimages = [None] * len(self.boxes)

    # def if_box_format(boxes):
    #     # the boxes format is List[List[float, float, float, float]]
    #     if isinstance(boxes, List):
    #         for box in boxes:
    #             if not (isinstance(box, List) and len(box) == 4):
    #                 return False

    #         return True
    #     else:
    #         return False



class SubImage(object):
    def __init__(self, path:str, cls_id:int, parent_image:str='', feat:np.ndarray=None) -> None:

        check_image_file(path)
        self.path: str = path
        self.cls_id:int = cls_id
        self.parent_image: str = parent_image
        self.feat: np.ndarray = feat


class DataStorage(object):
    def __init__(self) -> None:
        self.mainimgs = {}
        self.subimgs = {}

    def load_datafolder(self, path):
        self.mainimgs, self.subimgs = load_image_folder(path)

    # def load_image(self, path, label=None, project=''):
    #     idx = len(self.images) # self-increment index
    #     self.images[idx] = MainImage(path)

    # def load_imagefolder(path):
    #     pass

    # def load_subimage():
    #     pass

    # change class of one sub-image
    def set_subimg_cls(self, subimg_id, new_cls):
        self.subimgs[subimg_id].cls_id = new_cls

    def set_subimg_feat(self, subimg_id, feat):
        self.subimgs[subimg_id].feat = feat
        



def check_image_file(path):
    assert isfile(path), 'The path is not a regular file'
    filename = basename(path)
    assert filename.lower().endswith(IMG_EXTENSIONS), 'the file is not image'

def load_image_folder(path:str):
    # Scan the directory path and find all images
    # For MainImage, if it has label file, the label path is the same as image,
    # but with .txt suffix and yolo-format(each line is [id x y w h(percent)])
    #
    # The directory structure is:
    # --- directory
    #   |-- id_name_map.json
    #   |-- confirmed_subimg.json
    #   |-- sub_images
    #   |   |--[cls] # class name
    #   |   |   |--[image_id].jpg # sub image (classified)
    #   |   |   |--[image_id].pkl # sub image feature (classified)
    #   |   |-- [image_id].jpg # sub image (unclassified)
    #   |   |-- [image_id].pkl # sub image feature (unclassified)
    #   |-- preview
    #   |   |--[image_preview].jpg # detection result
    #   |-- [image].jpg # main image
    #   |-- [image].txt # label

    assert isdir(path), 'The path is not a directory'
    id_name_map = {}
    confirmed_subimg = {}
    subimg_path = ''
    preview_path = ''
    images = {} # file id: file path
    labels = {} # file id: file path

    subimgs = {} # subimg id: file path
    # feats = {}

    # scan base directory
    for name in os.listdir(path):
        if name == 'id_name_map.json':
            with open(join(path, name), 'r') as f:
                id_name_map = json.load(f)

        elif name == 'confirmed_subimg.json':
            with open(join(path, name), 'r') as f:
                confirmed_subimg = json.load(f)

        elif name == 'sub_images':
            subimg_path = join(path, name)
            assert isdir(subimg_path), 'The sub_image should be a directory'

        elif name == 'preview':
            preview_path = join(path, name)
            assert isdir(preview_path), 'The sub_image should be a directory'

        elif name.lower().endswith('jpg'):
            images[splitext(name)[0]] = join(path, name)

        elif name.lower().endswith('txt'):
            labels[splitext(name)[0]] = join(path, name)

    # convert value type, file id: MainImage object
    for k in images.keys():
        images[k] = MainImage(images[k])

    # scan preview directory
    if preview_path:
        for name in os.listdir(preview_path):
            if name.lower().endswith('.jpg'):
                main_image_id = splitext(name)[0].split('_')[0]
                images[main_image_id].preview_path = join(preview_path, name)


    # scan subimg directory
    if subimg_path:
        imgs = load_subimg_folder(subimg_path, cls_id=-1)
        subimgs.update(imgs)

        for name in os.listdir(subimg_path):
            dir_path = join(subimg_path, name)
            if isdir(dir_path):
                imgs = load_subimg_folder(dir_path, cls_id=name)
                subimgs.update(imgs)


    # add labels to MainImage
    for img_id in images.keys():
        if img_id in labels:
            label_path = labels[img_id]
            boxes = load_label_file(label_path)
            images[img_id].boxes = boxes
            images[img_id].subimages = [None] * len(boxes) # reset

    # add subimgs to MainImage
    # based on main image (if there is only subimage, ignore it)
    # merge all sub-images to main image
    main_sub_map = {} # {mainimg_id:[subimg_id, box_id]}
    for subimg_id in subimgs.keys():
        names = subimg_id.split('_')
        if len(names) == 2:
            mainimg_id, box_id = names
        else:
            continue # ignore
        box_id = int(box_id)
        if mainimg_id in main_sub_map:
            main_sub_map[mainimg_id].append([subimg_id, box_id])
        else:
            main_sub_map[mainimg_id] = [subimg_id, box_id]

    for mainimg_id, mainimg in images.items():
        if mainimg_id in main_sub_map:
            # modify MainImage object
            subimg_id, box_id = main_sub_map[mainimg_id]
            mainimg.subimages[box_id] = subimgs[subimg_id]
            # modify SubImage object
            subimgs[subimg_id].parent_image = mainimg_id

    return images, subimgs

def load_label_file(path):
    # load yolo-format label txt file
    with open(path, 'r') as f:
        labels = [line.strip().split() for line in f.readlines() if len(line)]
    boxes = []
    for label in labels:
        assert len(label)==5, f'wrong format of label file {path}'
        xywh = np.array(label[1:], dtype=np.float32)
        boxes.append(Box(*xywh, cls=int(label[0])))
    return boxes


def load_subimg_folder(path, cls_id) -> Dict[str, SubImage]:
    # only load .jpg and .pkl files in one directory, not include sub-folders
    # based on .jpg files (if there is only feature of subimage, ignore it)
    imgs = {}
    pkls = {}
    for name in os.listdir(path):
        subimg_id = splitext(name)[0]
        fullpath = join(path, name)
        if name.lower().endswith('.jpg'):
            imgs[subimg_id] = fullpath
        elif name.lower().endswith('.pkl'):
            pkls[subimg_id] = fullpath

    # merge two type files
    subimgs = {} # subimg id : SubImage object
    for subimg_id in imgs.keys():
        feat = None
        if subimg_id in pkls:
            with open(pkls[subimg_id], 'rb') as f:
                feat = pickle.load(f)
        subimgs[subimg_id] = SubImage(path=imgs[subimg_id], cls_id=cls_id, feat=feat)

    return subimgs


    

if __name__ == '__main__':
    folder_path = '/Users/jiahua/Downloads/moving_det_cv/collect2_data/capture_labelled/IPS_2022-03-26.17.24.00.8020'
    load_image_folder(folder_path)