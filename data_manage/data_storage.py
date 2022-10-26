from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
from collections import OrderedDict
import os
from os.path import join, basename, isdir, isfile, splitext
from PIL.Image import Image as PILImage
from PIL import ImageDraw
import json
import pickle
import shutil



class Box():
    def __init__(self, x, y, w, h, width=None, height=None, cls=-1) -> None:
        '''
        If width and height is not passed, the four numbers ahead is treated as
        xywh, each of them is within (0,1); if passed, use x1y1x2y2 format and
        four numbers are the actual coordinates of the box in the image.
        '''
        # -1 means unknown class
        if (width is not None) and (height is not None):
            # treat as xyxy format
            self.x1, self.y1, self.x2, self.y2  = x, y, w, h
            self.x, self.y, self.w, self.h = xyxy2xywh(self.x1, self.y1, self.x2, self.y2, width, height)
            self.width, self.height = width, height
        else:
            # treat as xywh format
            self.x, self.y, self.w, self.h = x, y, w, h
            self.x1, self.y1, self.x2, self.y2 = None, None, None, None
            self.width, self.height = None, None

        self.cls: int = cls

    def to_str(self):
        return f'{self.cls} {self.x:g} {self.y:g} {self.w:g} {self.h:g}'

    def get_xywh(self):
        return self.x, self.y, self.w, self.h

    def get_xyxy(self, width, height):
        if self.x1 is None:
            self.width, self.height = width, height
            self.x1, self.y1, self.x2, self.y2 = xywh2xyxy(self.x, self.y, self.w, self.h, width, height)

        return self.x1, self.y1, self.x2, self.y2



class SubImage(object):
    def __init__(self, path:str, cls_id:int, mainimg_id:str='', feat:np.ndarray=None) -> None:

        check_image_file(path)
        self.path: str = path
        self.id = os.path.splitext(os.path.basename(self.path))[0]
        self.feat_path: str = self.path.replace('.jpg', '.pkl')
        self.feat: np.ndarray = feat
        self.cls_id:int = cls_id

        mainimg_id, box_id = self.id.rsplit('_', maxsplit=1)
        self.mainimg_id: str = mainimg_id
        self.confirmed: bool = False

    def delete_files(self):
        os.remove(self.path)
        if os.path.exists(self.feat_path):
            os.remove(self.feat_path)
        print(f'sub image {os.path.basename(self.path)} has been deleted')

    def get_feat(self, extractor):
        # the extractor is an object having `encode()` method, which can encode
        # a PILImage into a feat in form of numpy.ndarray
        if self.feat is None:
            img = Image.open(self.path)
            self.feat = extractor.encode([img])[0]
            with open(self.feat_path, 'wb') as f:
                pickle.dump(self.feat, f)

        return self.feat


class MainImage(object):
    def __init__(self, path, preview_path=None, boxes=None, subimgs=None) -> None:
        check_image_file(path)
        self.path: str = path
        self.dir = os.path.dirname(self.path)
        self.id = os.path.splitext(os.path.basename(self.path))[0]
        self.info_path: str = self.path.replace('.jpg', '.json')
        self.label_path: str = self.path.replace('.jpg', '.txt')
        self.preview_path = os.path.join(self.dir, 'preview', f'{self.id}_preview.jpg')

        self.boxes: List[Box] = boxes if boxes else list()

        if subimgs:
            assert len(subimgs) == len(boxes), 'number of subimages and boxes not match'
            self.subimgs = OrderedDict(subimgs)
        else:
            self.subimgs = OrderedDict()


    def save_label(self):
        label_str = ''.join([box.to_str() + '\n' for box in self.boxes if box])
        with open(self.label_path, 'w') as f:
            f.write(label_str)

    def save_info(self):
        # save its subimage ids and their confirm status to info files
        assert len(self.subimgs) == len(self.boxes), 'number of subimages and boxes not match'
        infos = []
        for subimg_id, subimg in self.subimgs.items():
            infos.append([subimg_id, subimg.confirmed])
        with open(self.info_path, 'w') as f:
            json.dump(infos, f)

    # def gen_info(self):
    #     # generate info file based on boxes/label file
    #     self.subimg_infos = [[f'{self.id}_{i}', False] for i, _ in enumerate(self.boxes)]
    #     self.save_info()

    def save_preview(self):
        os.makedirs(os.path.join(self.dir, 'preview'), exist_ok=True)
        img = Image.open(self.path)
        width, height = img.size
        draw = ImageDraw.Draw(img)
        box: Box
        for box in self.boxes:
            x1, y1, x2, y2 = xywh2xyxy(box.x, box.y, box.w, box.h, width, height)
            draw.rectangle((x1,y1,x2,y2), outline='red', width=9)

        img.save(self.preview_path)

    def del_subimg(self, subimg_id_del: str):
        # delete found sub image id and update the label/info/preview files
        for i, subimg_id in enumerate(self.subimgs):
            if subimg_id_del == subimg_id:
                del self.boxes[i]
                del self.subimgs[subimg_id]
                self.save_label()
                self.save_info()
                self.save_preview()
                break


    def add_boxes(self, boxes: List[Box]):
        old_len = len(self.boxes)
        # add boxes to main image
        self.boxes.extend(boxes)
        self.save_label()
        self.save_preview()

        # add corresponding sub images
        subimgs: Dict[str:SubImage] = {}

        dir_path = os.path.join(self.dir, 'sub_images')
        os.makedirs(dir_path, exist_ok=True)
        img = Image.open(self.path)
        width, height = img.size

        for i in range(old_len, len(self.boxes)):
            box = self.boxes[i]

            # add subimage id to main image, avoid duplicate ids
            box_id = i
            while True:
                subimg_id = f'{self.id}_{box_id}'
                if subimg_id in self.subimgs:
                    box_id += 1  # id exists, try new box_id
                else:
                    break # id not exists, can be used

            # crop the subimage
            x1, y1, x2, y2 = box.get_xyxy(width, height)
            subimg_img = img.crop([x1, y1, x2, y2])

            # save to disk
            subimg_path = os.path.join(dir_path, f'{subimg_id}.jpg')
            subimg_img.save(subimg_path)

            # create SubImage object to return
            subimg = SubImage(subimg_path, cls_id=-1, mainimg_id=self.path)
            subimgs[subimg_id] = subimg
            self.subimgs[subimg_id] = subimg

        return subimgs

    def delete_files(self):
        os.remove(self.path)
        if os.path.exists(self.label_path):
            os.remove(self.label_path)
        if os.path.exists(self.preview_path):
            os.remove(self.preview_path)
        if os.path.exists(self.info_path):
            os.remove(self.info_path) 

        print(f'main image {os.path.basename(self.path)} has been deleted')

    def reset(self):
        self.boxes = []
        self.subimgs = OrderedDict()
        self.save_label()
        self.save_info()
        self.save_preview()


class DataStorage(object):
    def __init__(self) -> None:
        self.root: str = '.'
        self.mainimgs: dict[str, MainImage] = {}
        self.subimgs: dict[str, SubImage] = {}
        self.id2name: dict[int, str] = {}
        self.name2id: dict[str, int] = {}

    def load_datafolder(self, path, force=False):
        if path != self.root or force:
            self.mainimgs, self.subimgs, self.id2name = load_image_folder(path)
            self.name2id = {v: k for k, v in self.id2name.items()}
            self.root = path

    # change class of one sub-image
    def set_subimg_cls(self, subimg_id:str, new_cls_id:int, confirm=True):
        subimg: SubImage = self.subimgs[subimg_id]
        if new_cls_id != subimg.cls_id:
            subimg.cls_id = new_cls_id
            # move the subimg/feat file to new location
            subimg_folder = join(self.root, 'sub_images')
            if new_cls_id == -1:
                dst = join(subimg_folder, basename(subimg.path))
            else:
                new_cls_name = self.id2name[new_cls_id]
                os.makedirs(join(subimg_folder, new_cls_name), exist_ok=True)
                dst = join(subimg_folder, new_cls_name, basename(subimg.path))

            shutil.move(src=subimg.path, dst=dst)
            if subimg.feat is not None:
                if os.path.exists(subimg.feat_path):
                    shutil.move(src=subimg.feat_path, dst=dst.replace('.jpg', '.pkl'))
            subimg.path = dst

            # change boxes of its main image
            mainimg: MainImage = self.mainimgs[subimg.mainimg_id]
            for i, (id, _) in enumerate(mainimg.subimgs.items()):
                if subimg_id == id:
                    mainimg.boxes[i].cls = new_cls_id
                    break

            # overwrite the label/info file
            mainimg.save_label()
            mainimg.save_info()

        # add to confirmed list, and save it
        if confirm:
            self.confirm_subimgs(subimg_id)


    def confirm_subimgs(self, ids:Tuple[str,List[str]]):
        # accept single id or list of ids
        if isinstance(ids, str):
            ids = [ids]

        mainimg_ids = []
        for subimg_id in ids:
            subimg: SubImage = self.subimgs[subimg_id]
            if subimg.cls_id != -1: # ignore confirm if no class name set
                subimg.confirmed = True
                mainimg_ids.append(subimg.mainimg_id)

        # update info files of the main images
        mainimg_ids = list(set(mainimg_ids))
        for mainimg_id in mainimg_ids:
            mainimg: MainImage = self.mainimgs[mainimg_id]
            mainimg.save_info()

    def add_mainimg(self, image:PILImage, mainimg_id):
        name = f'{mainimg_id}.jpg'
        path = join(self.root, name)
        image.save(path)
        self.mainimgs[mainimg_id] = MainImage(path=path)

    # TODO: change boxes position of mainimg

    def delete_mainimg(self, mainimg_id):
        mainimg = self.mainimgs[mainimg_id]
        # delete corresponding subimgs
        for subimg_id in mainimg.subimgs.keys():
            subimg:SubImage = self.subimgs[subimg_id]
            subimg.delete_files()
            del self.subimgs[subimg_id]

        mainimg.delete_files()
        del self.mainimgs[mainimg_id]

    def reset_mainimg(self, mainimg_id):
        mainimg = self.mainimgs[mainimg_id]
        # delete corresponding subimgs
        for subimg_id in mainimg.subimgs.keys():
            subimg:SubImage = self.subimgs[subimg_id]
            subimg.delete_files()
            del self.subimgs[subimg_id]

        mainimg.reset()


    def delete_subimgs(self, ids:Tuple[str,List[str]]):
        # accept single id or list of ids
        if isinstance(ids, str):
            ids = [ids]

        for subimg_id in ids:
            subimg:SubImage = self.subimgs[subimg_id]
            subimg.delete_files()
            # delete the record in its main image
            mainimg: MainImage = self.mainimgs[subimg.mainimg_id]
            mainimg.del_subimg(subimg_id)
            # delete sub image record
            del self.subimgs[subimg_id]


    def add_cls(self, new_cls_name:str):
        if new_cls_name in self.name2id:
            return
        new_cls_id = len(self.id2name)
        self.id2name[new_cls_id] = new_cls_name
        self.name2id[new_cls_name] = new_cls_id

        # save to id_name_map.json
        # dump to string first to avoid error during writing, causing the
        # origin json file empty
        json_str = json.dumps(self.id2name, indent=4, ensure_ascii=False)
        with open(join(self.root, 'id_name_map.json'), 'w') as f:
            f.write(json_str)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def check_image_file(path):
    assert isfile(path), 'The path is not a regular file'
    filename = basename(path)
    assert filename.lower().endswith(IMG_EXTENSIONS), 'the file is not image'

def load_image_folder(path:str):
    # Scan the directory path and find all images
    # For MainImage, if it has label file, the label path is the same as image,
    # with .txt suffix and yolo-format(each line is [id x y w h(percent)])
    #
    # The directory structure is:
    # --- root directory
    #   |-- id_name_map.json # convert class id to class name
    #   |-- sub_images
    #   |   |-- [cls_name] # class name
    #   |   |    |--[subimg_id].jpg # sub image (classified)
    #   |   |    |--[subimg_id].pkl # CNN feature of sub image (classified)
    #   |   |-- [subimg_id].jpg # sub image (unclassified)
    #   |   |-- [subimg_id].pkl # sub image feature (unclassified)

    #   |-- preview
    #   |   |--[image_preview].jpg # visualized detection result
    #   |-- [mainimg_id].jpg # main image. The necessary file, other files are all based on it.
    #   |-- [mainimg_id].txt # label of main image (yolo format)
    #   |-- [mainimg_id].json # the subimg_ids of mainimg and their comfirming status

    # The most basic files are main images, and then the label files.
    # Preview/info/sub-images are all depended on label files.

    assert isdir(path), 'The path is not a directory'
    id2name, name2id = {}, {}
    subimg_folder = ''
    preview_folder = ''
    # mainimg_previews = {} # file id: image jpg path
    mainimg_labels = {} # file id: label txt path
    mainimg_infos = {} # file id: info json path

    mainimgs = {}# mainimg id: MainImage
    subimgs = {} # subimg id: SubImage

    # scan root directory
    for name in os.listdir(path):
        if name == 'id_name_map.json':
            with open(join(path, name), 'r') as f:
                id2name = {int(k):v for k,v in json.load(f).items()}
                name2id = {v:k for k,v in id2name.items()}

        elif name == 'sub_images':
            subimg_folder = join(path, name)
            assert isdir(subimg_folder), f'{subimg_folder} is not a directory'

        elif name == 'preview':
            preview_folder = join(path, name)
            assert isdir(preview_folder), f'{preview_folder} is not a directory'

        elif name.lower().endswith('jpg'):
            mainimg_id = splitext(name)[0]
            mainimgs[mainimg_id] = MainImage(join(path, name))

        elif name.lower().endswith('txt'):
            mainimg_id = splitext(name)[0]
            mainimg_labels[mainimg_id] = join(path, name)

        elif name.lower().endswith('json'):
            mainimg_id = splitext(name)[0]
            mainimg_infos[mainimg_id] = join(path, name)

    ### files recording and validation
    # remove label files which don't have origin main images
    for mainimg_id in list(mainimg_labels.keys()):
        if mainimg_id not in mainimgs:
            os.remove(mainimg_labels[mainimg_id])
            del mainimg_labels[mainimg_id]

    # remove info files which don't have label files
    for mainimg_id in list(mainimg_infos.keys()):
        if mainimg_id not in mainimg_labels:
            os.remove(mainimg_infos[mainimg_id])
            del mainimg_infos[mainimg_id]

    # scan preview directory
    if preview_folder:
        for name in os.listdir(preview_folder):
            if name.lower().endswith('.jpg'):
                mainimg_id = splitext(name)[0].replace('_preview','')
                preview_path = join(preview_folder, name)
                # remove the preview files which don't have label files
                if mainimg_id not in mainimg_labels:
                    os.remove(preview_path)


    # scan subimg directory
    if subimg_folder:
        # load unclassified subimages
        imgs: Dict[str, SubImage] = load_subimg_folder(subimg_folder, cls_id=-1)
        subimgs.update(imgs)

        # load classified subimages
        for name in os.listdir(subimg_folder):
            dir_path = join(subimg_folder, name)
            if isdir(dir_path):
                imgs: Dict[str, SubImage] = load_subimg_folder(dir_path, cls_id=name2id[name])
                subimgs.update(imgs)

        # remove sub images which don't have label files
        for subimg_id in list(subimgs.keys()):
            subimg: SubImage = subimgs[subimg_id]
            if subimg.mainimg_id not in mainimg_labels:
                subimg.delete_files()
                del subimgs[subimg_id]

    ### merging data
    # add labels, infos and subimgs to MainImage
    main_sub_map = {}  # {mainimg_id:List[subimg_id]}
    for subimg_id, subimg in subimgs.items():
        if subimg.mainimg_id in main_sub_map:
            main_sub_map[subimg.mainimg_id].append(subimg_id)
        else:
            main_sub_map[subimg.mainimg_id] = [subimg_id]

    for mainimg_id, label_path in mainimg_labels.items():
        boxes: List[Box] = load_label_file(label_path)

        all_match = False
        if mainimg_id in mainimg_infos:
            info_path = mainimg_infos[mainimg_id]
            infos = load_info_file(info_path)
            # check info length and sub images files completed
            if (len(boxes) == len(infos)) and (
                len(boxes) == len(main_sub_map[mainimg_id])):
                all_match = True

        # connect main image with label/info/sub images
        mainimg: MainImage = mainimgs[mainimg_id]
        if all_match:
            mainimg.boxes = boxes
            for subimg_id, confirmed in infos:
                subimg: SubImage = subimgs[subimg_id]
                subimg.confirmed = confirmed # update confirm status
                mainimg.subimgs[subimg_id] = subimg # register to main image
        else:
            # remove not completed sub images firstly
            if mainimg_id in main_sub_map:
                for subimg_id in main_sub_map[mainimg_id]:
                    subimg:SubImage = subimgs[subimg_id]
                    subimg.delete_files()
                    del subimgs[subimg_id]

            # generate new info files, preview and subimages
            new_subimgs: Dict[str:SubImage] = mainimg.add_boxes(boxes)
            mainimg.save_info()
            mainimg.save_preview()

            subimgs.update(new_subimgs) # add new sub images

    return mainimgs, subimgs, id2name

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

def load_info_file(path):
    # a list of 2-elements list, stored/loaded by json
    # [[subimg_id1, confirm_state], [subimg_id2, confirm_state], ...]
    with open(path, 'r') as f:
        infos = json.load(f)
    return infos


def load_subimg_folder(path, cls_id) -> Dict[str, SubImage]:
    # only load .jpg and .pkl files in one directory, not include sub-folders
    # based on .jpg files (if there is only feature of subimage, ignore it)
    imgs = {}
    pkls = {}
    for name in os.listdir(path):
        subimg_id = splitext(name)[0]
        full_path = join(path, name)
        if name.lower().endswith('.jpg'):
            imgs[subimg_id] = full_path
        elif name.lower().endswith('.pkl'):
            pkls[subimg_id] = full_path

    # merge two type files
    subimgs = {} # subimg id : SubImage object
    for subimg_id in imgs.keys():
        feat = None
        if subimg_id in pkls:
            with open(pkls[subimg_id], 'rb') as f:
                feat = pickle.load(f)
            del pkls[subimg_id] # remove the loaded feat paths
        subimgs[subimg_id] = SubImage(path=imgs[subimg_id], cls_id=cls_id, feat=feat)

    # For the feature files whose the origin subimg not exists, remove them
    for full_path in pkls.values():
        os.remove(full_path)

    return subimgs


def xyxy2xywh(x1, y1, x2, y2, width, height) -> Tuple[float, float, float, float]:
    x = (x1 + x2) / 2 / width
    y = (y1 + y2) / 2 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return x, y, w, h


def xywh2xyxy(x, y, w, h, width, height):
    x1 = (x - w / 2) * width
    x2 = (x + w / 2) * width
    y1 = (y - h / 2) * height
    y2 = (y + h / 2) * height
    return x1, y1, x2, y2


if __name__ == '__main__':
    folder_path = '/Users/jiahua/Downloads/moving_det_cv/collect2_data/capture_labelled/IPS_2022-03-26.17.24.00.8020'
    load_image_folder(folder_path)