from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
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
        if width is not None and height is not None:
            # treat as xyxy format
            self.x1, self.y1, self.x2, self.y2  = x, y, w, h
            self.x, self.y, self.w, self.h = xyxy2xywh(self.x1, self.y1, self.x2, self.y2, width, height)
            self.width, self.height = width, height
        else:
            # treat as xywh format
            self.x, self.y, self.w, self.h = x, y, w, h

        self.cls: int = cls

    def to_str(self):
        return f'{self.cls} {self.x:g} {self.y:g} {self.w:g} {self.h:g}'

    def get_xywh(self):
        return self.x, self.y, self.w, self.h

    def get_xyxy(self, width, height):
        if self.width is not None:
            self.width, self.height = width, height
            self.x1, self.y1, self.x2, self.y2 = xywh2xyxy(self.x, self.y, self.w, self.h, width, height)

        return self.x1, self.y1, self.x2, self.y2



class SubImage(object):
    def __init__(self, path:str, cls_id:int, parent_image:str='', feat:np.ndarray=None) -> None:

        check_image_file(path)
        self.path: str = path
        self.cls_id:int = cls_id
        self.parent_image: str = parent_image
        self.feat: np.ndarray = feat

    def delete_files(self):
        os.remove(self.path)
        feat_path = self.path.replace('.jpg','.pkl')
        if self.feat is not None and os.path.exists(feat_path):
            os.remove(feat_path)
        print(f'sub image {os.path.basename(self.path)} has been deleted')

    def get_feat(self, extractor):
        # the extractor is an object having `encode()` method, which can encode
        # a PILImage into a feat in form of numpy.ndarray
        if self.feat is None:
            img = Image.open(self.path)
            self.feat = extractor.encode([img])[0]
            with open(self.path.replace('.jpg', '.pkl'), 'wb') as f:
                pickle.dump(self.feat, f)

        return self.feat
        

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


        self.id = os.path.basename(self.path).replace('.jpg','')
        self.dir = os.path.dirname(self.path)

    # def if_box_format(boxes):
    #     # the boxes format is List[List[float, float, float, float]]
    #     if isinstance(boxes, List):
    #         for box in boxes:
    #             if not (isinstance(box, List) and len(box) == 4):
    #                 return False

    #         return True
    #     else:
    #         return False

    def save_label(self):
        label_path = self.path.replace('.jpg', '.txt')
        label_str = ''.join([box.to_str() + '\n' for box in self.boxes if box])
        with open(label_path, 'w') as f:
            f.write(label_str)

    def save_preview(self):
        dir_path = os.path.join(self.dir, 'preview')
        os.makedirs(dir_path, exist_ok=True)
        img = Image.open(self.path)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        box: Box
        for box in self.boxes:
            x1, y1, x2, y2 = xywh2xyxy(box.x, box.y, box.w, box.h, width, height)
            draw.rectangle((x1,y1,x2,y2), outline='red', width=5)

        self.preview_path =  os.path.join(dir_path, f'{self.id}_preview.jpg')
        img.save(self.preview_path)

    def del_subimg(self, subimg_id_del:str):
        idx_to_del:int = self.subimages.index(subimg_id_del)
        del self.subimages[idx_to_del]
        del self.boxes[idx_to_del]

        self.save_label()
        self.save_preview()

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

            # add subimage id to main image
            subimg_id = f'{self.id}_{i}'
            self.subimages.append(subimg_id)

            # crop the subimage
            x1, y1, x2, y2 = box.get_xyxy(width, height)
            subimg_img = img.crop([x1, y1, x2, y2])

            # save to disk
            subimg_path = os.path.join(dir_path, f'{subimg_id}.jpg')
            subimg_img.save(subimg_path)

            # create SubImage object to return
            subimg = SubImage(subimg_path, cls_id=-1, parent_image=self.path)
            subimgs[subimg_id] = subimg

        return subimgs

    def delete_files(self):
        os.remove(self.path)
        label_path = self.path.replace('.jpg', '.txt')
        if os.path.exists(label_path):
            os.remove(label_path)
        if self.preview_path and os.path.exists(self.preview_path):
            os.remove(self.preview_path)

        print(f'main image {os.path.basename(self.path)} has been deleted')



class DataStorage(object):
    def __init__(self) -> None:
        self.root: str = '.'
        self.mainimgs: dict[str, MainImage] = {}
        self.subimgs: dict[str, SubImage] = {}
        self.id2name: dict[int, str] = {}
        self.name2id: dict[str, int] = {}
        self.confirmed_subimg: dict[str, str] = {}

    def load_datafolder(self, path, force=False):
        if path != self.root or force:
            self.mainimgs, self.subimgs, self.id2name, self.confirmed_subimg = load_image_folder(path)
            self.name2id = {v: k for k, v in self.id2name.items()}
            self.root = path

    # def load_image(self, path, label=None, project=''):
    #     idx = len(self.images) # self-increment index
    #     self.images[idx] = MainImage(path)



    # change class of one sub-image
    def set_subimg_cls(self, key:str, new_cls_id:int, confirm=True):
        subimg:SubImage = self.subimgs[key]
        if new_cls_id != subimg.cls_id:
            subimg.cls_id = new_cls_id
            # move the subimg file to new location
            subimg_folder = join(self.root, 'sub_images')
            if new_cls_id == -1:
                dst = join(subimg_folder, basename(subimg.path))
            else:
                new_cls_name = self.id2name[new_cls_id]
                os.makedirs(join(subimg_folder, new_cls_name), exist_ok=True)
                dst = join(subimg_folder, new_cls_name, basename(subimg.path))

            shutil.move(src=subimg.path, dst=dst)
            if subimg.feat is not None:
                feat_path = subimg.path.replace('.jpg', '.pkl')
                if os.path.exists(feat_path):
                    shutil.move(src=feat_path, dst=dst.replace('.jpg', '.pkl'))
            subimg.path = dst



            # change its parent image object
            if subimg.parent_image:
                mainimg = self.mainimgs[subimg.parent_image]
                for i, subimg_key in enumerate(mainimg.subimages):
                    if key == subimg_key:
                        mainimg.boxes[i].cls = new_cls_id
                        break
                # overwrite the label file
                mainimg.save_label()

        # add to confirmed list, and save it
        if confirm:
            self.confirm_subimgs(key)

    def confirm_subimgs(self, keys:Tuple[str,List[str]]):
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            if self.subimgs[key].cls_id == -1:
                continue
            self.confirmed_subimg[key] = '1'

        with open(join(self.root, 'confirmed_subimg.json'), 'w') as f:
            json.dump(self.confirmed_subimg, f,
                    indent=4,
                    ensure_ascii=False)

    def save_confirm(self):
        with open(join(self.root, 'confirmed_subimg.json'), 'w') as f:
            json.dump(self.confirmed_subimg, f,
                      indent=4,
                      ensure_ascii=False)


    # TODO: change boxes position of mainimg

    def delete_mainimg(self, key):
        mainimg = self.mainimgs[key]
        # delete corresponding subimgs
        self.delete_subimgs(mainimg.subimages[::-1])

        mainimg.delete_files()
        del self.mainimgs[key]

    def delete_subimgs(self, keys):
        if isinstance(keys, str):
            keys = [keys]

        for i, key in enumerate(keys):
            subimg = self.subimgs[key]
            # delete sub image file
            subimg.delete_files()

            if subimg.parent_image:
                # delete its record in main image
                mainimg: MainImage = self.mainimgs[subimg.parent_image]
                mainimg.del_subimg(key)

                # re-order box_id of the left subimages
                # rename sub images
                old2new = {}
                for i, old_id in enumerate(mainimg.subimages):
                    new_id = f'{subimg.parent_image}_{i}'
                    old2new[old_id] = new_id
                    mainimg.subimages[i] = new_id

                # apply this change to subimgs dict
                for old_id, new_id in old2new.items():
                    if old_id == new_id: continue
                    # rename the key of sub image object
                    subimg: SubImage = self.subimgs.pop(old_id)
                    self.subimgs[new_id] = subimg

                    # rename the re-ordered sub image+feat files
                    new_path = subimg.path.replace(old_id, new_id)
                    shutil.move(src=subimg.path, dst=new_path)

                    if subimg.feat is not None:
                        feat_path = subimg.path.replace('.jpg', '.pkl')
                        if os.path.exists(feat_path):
                            shutil.move(src=feat_path, dst=feat_path.replace(old_id, new_id))
                    subimg.path = new_path

                    # change the subimg id in confirmed list
                    if old_id in self.confirmed_subimg:
                        del self.confirmed_subimg[old_id]
                        self.confirmed_subimg[new_id] = '1'

                    # this may influence other keys in list to delete, so update them
                    for j, key_j in enumerate(keys[i+1:]):
                        if key_j == old_id:
                            keys[i+1+j] = new_id
                            break

            del self.subimgs[key]
            if key in self.confirmed_subimg:
                del self.confirmed_subimg[key]
                self.save_confirm()



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
    # but with .txt suffix and yolo-format(each line is [id x y w h(percent)])
    #
    # The directory structure is:
    # --- directory
    #   |-- id_name_map.json
    #   |-- confirmed_subimg.json
    #   |-- sub_images
    #   |   |--[cls_name] # class name
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
                id_name_map = {int(k):v for k,v in json.load(f).items()}
                name2id = {v:k for k,v in id_name_map.items()}

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
                mainimg_key = splitext(name)[0].replace('_preview','')
                images[mainimg_key].preview_path = join(preview_path, name)


    # scan subimg directory
    if subimg_path:
        imgs = load_subimg_folder(subimg_path, cls_id=-1)
        subimgs.update(imgs)

        for name in os.listdir(subimg_path):
            dir_path = join(subimg_path, name)
            if isdir(dir_path):
                imgs = load_subimg_folder(dir_path, cls_id=name2id[name])
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
    main_sub_map = {} # {mainimg_id:List[[subimg_id, box_id]]}
    for subimg_id in subimgs.keys():
        names = subimg_id.rsplit('_', maxsplit=1)
        if len(names) == 2:
            mainimg_id, box_id = names
        else:
            continue # ignore
        box_id = int(box_id)
        if mainimg_id in main_sub_map:
            main_sub_map[mainimg_id].append([subimg_id, box_id])
        else:
            main_sub_map[mainimg_id] = [[subimg_id, box_id]]

    mainimg_id: str; mainimg: MainImage
    for mainimg_id, mainimg in images.items():
        if mainimg_id in main_sub_map:
            # modify MainImage object
            for subimg_id, box_id in main_sub_map[mainimg_id]:
                mainimg.subimages[box_id] = subimg_id
                # modify SubImage object
                subimgs[subimg_id].parent_image = mainimg_id

    return images, subimgs, id_name_map, confirmed_subimg

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